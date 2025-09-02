import argparse
import time
from typing import Optional

import jax
from flax import nnx

from jaxrl.experiment import Experiment
from jaxrl.envs.env_config import create_env


def block_all(xs):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark JAX environments.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=str, help="Path to a config JSON file.")
    src.add_argument(
        "--run",
        type=str,
        help="Existing experiment run token (loads config from results/).",
    )

    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        help="Select a specific env when using a multi env config.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for RNGs and actions."
    )
    parser.add_argument(
        "--vec",
        type=int,
        default=1,
        help="Number of vectorized env copies (VectorWrapper).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Steps per rollout (defaults to config.max_env_steps).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup rollouts before timing.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of timed rollouts to average.",
    )
    return parser.parse_args()


def main(
    config_path: Optional[str],
    run_token: Optional[str],
    seed: int,
    vec_count: int,
    selector: Optional[str],
    warmup: int,
    iters: int,
):
    # Load experiment configuration
    if run_token is not None:
        experiment = Experiment.load(run_token, base_dir="results")
    else:
        experiment = Experiment.from_config_file(config_path, base_dir="", create_directories=False)

    max_steps = experiment.config.max_env_steps

    # Create environment
    env = create_env(
        experiment.config.environment, max_steps, vec_count=vec_count, selector=selector
    )

    if not env.is_jittable:
        raise ValueError("Selected environment is not JIT compatible (is_jittable=False)")

    rngs = nnx.Rngs(default=seed)

    # Capture constants used in JIT
    num_actions = env.action_spec.num_actions
    num_agents = env.num_agents

    @nnx.jit
    def reset_jit(key):
        return env.reset(key)

    @nnx.jit
    def rollout_jit(state, rng_key):
        def body(carry, _):
            state, rng = carry
            rng, akey, skey = jax.random.split(rng, 3)
            actions = jax.random.randint(akey, (num_agents,), minval=0, maxval=num_actions)
            state, ts = env.step(state, actions, skey)
            return (state, rng), ts

        (state, rng_key), last_ts = jax.lax.scan(body, (state, rng_key), None, length=max_steps)
        return state, last_ts

    # Warmup JIT: reset + one rollout a few times
    state, ts = reset_jit(rngs.env())
    block_all((state, ts))

    for _ in range(max(1, warmup)):
        state, ts = rollout_jit(state, rngs.env())
        block_all((state, ts))

    # Timed iterations
    times = []
    total_env_steps = max_steps * num_agents
    for _ in range(iters):
        t0 = time.perf_counter()
        state, ts = rollout_jit(state, rngs.env())
        # Ensure execution completes before stopping timer
        block_all((state, ts))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    steps_per_second = total_env_steps / avg_time

    print("Benchmark Results")
    print(f"- env: {type(env).__name__}")
    print(f"- vec_count: {vec_count}")
    print(f"- agents: {num_agents}")
    print(f"- warmup_rollouts: {warmup}")
    print(f"- timed_rollouts: {iters}")
    print(f"- avg_time_per_rollout: {avg_time:.6f}s")
    print(f"- env_steps_per_rollout: {total_env_steps}")
    print(f"- steps_per_second: {steps_per_second:,.2f}")


if __name__ == "__main__":
    args = parse_args()
    main(
        config_path=args.config,
        run_token=args.run,
        seed=args.seed,
        vec_count=args.vec,
        selector=args.selector,
        warmup=args.warmup,
        iters=args.iters,
    )
