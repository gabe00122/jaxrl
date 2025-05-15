import time
from jax import numpy as jnp, random, Array
from flax import nnx

from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.types import Observation


def main():
    rngs = nnx.Rngs(default=0)
    transformer = TransformerActorCritic(
        8,
        2,
        3,
        8,
        128,
        128,
        glu=False,
        gtrxl_gate=True,
        activation="relu",
        kernel_init=nnx.initializers.normal(),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=rngs,
    )

    iterations = 10
    batch_size = 144 * 2
    context_size = 1024
    transformer.create_kv_cache(batch_size, context_size, dtype=jnp.float32)

    print("Warming up...")
    for _ in range(10):
        model, rngs, action_buffer, value_buffer = bench(
            transformer, rngs, batch_size, context_size
        )

    action_buffer.block_until_ready()
    value_buffer.block_until_ready()
    print("Warmup finished.")

    print("Benchmarking...")
    start_time = time.time()

    for _ in range(iterations):
        model, rngs, action_buffer, value_buffer = bench(
            model, rngs, batch_size, context_size
        )

    action_buffer.block_until_ready()
    value_buffer.block_until_ready()
    end_time = time.time()
    print("Benchmarking finished.")

    duration = end_time - start_time
    total_actions = batch_size * context_size * iterations
    actions_per_second = total_actions / duration

    print(f"Execution time: {duration:.4f} seconds")
    print(f"Total actions: {total_actions}")
    print(f"Actions per second: {actions_per_second:.2f}")
    print(f"Action buffer shape: {action_buffer.shape}")
    print(f"Value buffer shape: {value_buffer.shape}")


@nnx.jit(donate_argnums=(0, 1), static_argnums=(2, 3))
def bench(
    model: TransformerActorCritic, rngs: nnx.Rngs, batch_size: int, context_size: int
):
    def body(i, input: tuple[TransformerActorCritic, nnx.Rngs, Array, Array]):
        model, rngs, ab, vb = input

        obs = Observation(
            agents_view=random.normal(
                rngs.default(), (batch_size, 1, 8), dtype=jnp.bfloat16
            ),
            time_steps=jnp.full((1, 1), i),
            last_action=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            last_reward=jnp.zeros((batch_size, 1), dtype=jnp.bfloat16),
            action_mask=None,
        )
        value, policy = model(obs, True)
        action = policy.sample(seed=rngs.default())

        ab = ab.at[:, i].set(jnp.squeeze(action))
        vb = vb.at[:, i].set(jnp.squeeze(value))

        return model, rngs, ab, vb

    action_buffer = jnp.zeros((batch_size, context_size), dtype=jnp.int32)
    value_buffer = jnp.zeros((batch_size, context_size), dtype=jnp.float32)

    model, rngs, action_buffer, value_buffer = nnx.fori_loop(
        0, context_size, body, (model, rngs, action_buffer, value_buffer)
    )
    return model, rngs, action_buffer, value_buffer


if __name__ == "__main__":
    main()
