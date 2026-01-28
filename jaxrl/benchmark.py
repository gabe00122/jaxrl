import time
from typing import Optional

import jax
import typer
from flax import nnx
from rich.console import Console

from jaxrl.experiment import Experiment
from mapox import create_env


console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


def block_all(xs):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)


def benchmark(
    config_path: Optional[str],
    run_token: Optional[str],
    seed: int,
    vec_count: int,
    env_name: Optional[str],
    steps: Optional[int],
    warmup: int,
    iters: int,
):
    # Load experiment configuration
    if run_token is not None:
        experiment = Experiment.load(run_token, base_dir="results")
    else:
        experiment = Experiment.from_config_file(config_path, base_dir="", create_directories=False)

    max_steps = steps or experiment.config.max_env_steps

    # Create environment
    env = create_env(
        experiment.config.environment,
        max_steps,
        vec_count=vec_count,
        env_name=env_name,
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
    def rollout_jit(rng_key):
        state, ts = reset_jit(rng_key)

        def body(carry, _):
            state, rng = carry
            rng, akey, skey = jax.random.split(rng, 3)
            actions = jax.random.randint(akey, (num_agents,), minval=0, maxval=num_actions)
            state, ts = env.step(state, actions, skey)
            return (state, rng), ts

        (state, rng_key), last_ts = jax.lax.scan(body, (state, rng_key), None, length=max_steps)
        return state, last_ts


    for _ in range(max(1, warmup)):
        state, ts = rollout_jit(rngs.env())
        block_all((state, ts))

    # Timed iterations
    times = []
    total_env_steps = max_steps * num_agents
    for _ in range(iters):
        t0 = time.perf_counter()
        state, ts = rollout_jit(rngs.env())
        # Ensure execution completes before stopping timer
        block_all((state, ts))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    steps_per_second = total_env_steps / avg_time

    console.print("Benchmark Results")
    console.print(f"- env: {type(env).__name__}")
    console.print(f"- vec_count: {vec_count}")
    console.print(f"- agents: {num_agents}")
    console.print(f"- warmup_rollouts: {warmup}")
    console.print(f"- timed_rollouts: {iters}")
    console.print(f"- avg_time_per_rollout: {avg_time:.6f}s")
    console.print(f"- env_steps_per_rollout: {total_env_steps}")
    console.print(f"- steps_per_second: {steps_per_second:,.2f}")


@app.command()
def main(
    config: Optional[str] = typer.Option(
        None, help="Path to a config JSON file.", rich_help_panel="Input"
    ),
    run: Optional[str] = typer.Option(
        None,
        help="Existing experiment run token (loads config from results/).",
        rich_help_panel="Input",
    ),
    env: Optional[str] = typer.Option(
        None, help="Select a specific env when using a multi env config."
    ),
    seed: int = typer.Option(0, help="Random seed for RNGs and actions."),
    vec: int = typer.Option(
        1, help="Number of vectorized env copies (VectorWrapper)."
    ),
    steps: Optional[int] = typer.Option(
        None, help="Steps per rollout (defaults to config.max_env_steps)."
    ),
    warmup: int = typer.Option(2, help="Number of warmup rollouts before timing."),
    iters: int = typer.Option(5, help="Number of timed rollouts to average."),
):
    if (config is None) == (run is None):
        raise typer.BadParameter("Provide exactly one of --config or --run")

    benchmark(config, run, seed, vec, env_name=env, steps=steps, warmup=warmup, iters=iters)


if __name__ == "__main__":
    app()
