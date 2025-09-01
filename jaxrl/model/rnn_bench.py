from functools import partial
from pathlib import Path
from time import perf_counter
import numpy as np
from jax import numpy as jnp
from flax import nnx
import jax

from jaxrl.config import load_config
from jaxrl.envs.specs import ObservationSpec
from jaxrl.model.network import TransformerActorCritic
from jaxrl.types import TimeStep


def _block_until_ready(out):
    # Ensure all device work is finished before measuring time
    leaves = jax.tree_util.tree_leaves(out)
    for x in leaves:
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()


def bench(fn, *args, n_warmup=1, n_iter=10, label=""):
    """
    Time a JAX function:
      - first_call_s: includes compilation (for jitted fns)
      - steady-state stats over n_iter executions
    """
    # First call (compile + run)
    t0 = perf_counter()
    out = fn(*args)
    _block_until_ready(out)
    first_call_s = perf_counter() - t0

    # Optional extra warmups (useful for scan variants, caches, etc.)
    for _ in range(max(0, n_warmup - 1)):
        out = fn(*args)
        _block_until_ready(out)

    # Steady-state runs
    times = []
    for _ in range(n_iter):
        t0 = perf_counter()
        out = fn(*args)
        _block_until_ready(out)
        times.append(perf_counter() - t0)

    times = np.array(times)
    print(f"\n[{label}]")
    print(f"  first call (compile+run): {first_call_s * 1e3:.2f} ms")
    print(f"  steady-state mean:        {times.mean() * 1e3:.2f} ms")
    print(f"  steady-state p50:         {np.percentile(times, 50) * 1e3:.2f} ms")
    print(f"  steady-state p95:         {np.percentile(times, 95) * 1e3:.2f} ms")
    print(f"  runs:                     {n_iter}")
    return times


def main():
    config = load_config(Path("./config/config.json").read_text())

    batch_size = 512
    obs_size = (5, 5)
    seq_length = 512

    rngs = nnx.Rngs(0)
    model = TransformerActorCritic(
        config.learner.model,
        ObservationSpec(jnp.int8, obs_size, 7),
        4,
        config.hl_gauss,
        seq_length,
        rngs=rngs,
    )

    time = jnp.repeat(
        jnp.arange(seq_length, dtype=jnp.int32)[None, :], batch_size, axis=0
    )

    timestep = TimeStep(
        obs=jnp.zeros((batch_size, seq_length, *obs_size), dtype=jnp.int8),
        time=time,
        last_action=jnp.zeros((batch_size, seq_length), dtype=jnp.int32),
        last_reward=jnp.zeros((batch_size, seq_length)),
        action_mask=None,
    )

    # --- define the two inference functions ---

    @partial(nnx.scan, in_axes=(1, nnx.Carry), out_axes=(1, nnx.Carry))
    def scanned_inference(ts, carry):
        # add a singleton time dimension so model expects [B, T, ...]
        ts = jax.tree_util.tree_map(lambda x: x[:, None, ...], ts)
        value, value_logits, policy, carry = model(ts, carry)
        value.squeeze(axis=1)
        return value, carry

    @jax.jit
    def layered_inference(model, timestep):
        value, value_logits, policy, carry = model(timestep)
        return value, carry

    # Initialize carry
    init_carry = model.initialize_carry(batch_size, rngs)

    # Touch both once so any lazy initialization happens before timing (optional)
    _block_until_ready(scanned_inference(timestep, init_carry))
    _block_until_ready(layered_inference(model, timestep))

    # --- benchmarks ---
    bench(
        layered_inference,
        model,
        timestep,
        n_warmup=1,
        n_iter=20,
        label="layered_inference (@jit)",
    )
    bench(
        scanned_inference,
        timestep,
        init_carry,
        n_warmup=1,
        n_iter=20,
        label="scanned_inference (nnx.scan)",
    )

    # If you want to see shapes once:
    v_scan, c_scan = scanned_inference(timestep, init_carry)
    _block_until_ready((v_scan, c_scan))
    print("\nscanned_inference value shape:", v_scan.shape)

    v_layer, c_layer = layered_inference(model, timestep)
    _block_until_ready((v_layer, c_layer))
    print("layered_inference value shape:", v_layer.shape)


if __name__ == "__main__":
    # (Optional) to avoid large preallocation on GPU:
    # import os; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
