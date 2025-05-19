import time
import jax
from jax import numpy as jnp, random, Array
from flax import nnx

from jaxrl.transformer.network import LinearObsEncoderConfig, TransformerActorCritic, TransformerBlockConfig, TransformerActorCriticConfig
from jaxrl.types import Observation

# num_layers: 12
# num_heads: 12
# d_model: 768
# ffn_size: 2048

def main():
    rngs = nnx.Rngs(default=0)
    
    config = TransformerActorCriticConfig(
        obs_encoder=LinearObsEncoderConfig(),
        hidden_features=768, #256,
        num_layers=12, #3,
        transformer_block=TransformerBlockConfig(
            ffn_size=2048, #512,
            glu=True,
            gtrxl_gate=False,
            max_seq_length=512,
            num_heads=12,
            gtrxl_bias=2.0,
        ),
        activation="gelu",
        dtype="bfloat16",
        param_dtype="float32",
        kernel_init="normal",
        norm="layer_norm",
    )
    # Initialize transformer with config and required modules
    transformer = TransformerActorCritic(config, 10, 10, rngs=rngs)

    iterations = 10
    batch_size = 32
    context_size = config.transformer_block.max_seq_length
    transformer.create_kv_cache(batch_size, context_size, dtype=config.dtype)

    print("Warming up...")
    model_def, model_state = nnx.split(transformer)
    rngs_def, rngs_state = nnx.split(rngs)
    action_buffer = jnp.zeros((batch_size, context_size), dtype=jnp.int32)
    value_buffer = jnp.zeros((batch_size, context_size), dtype=config.dtype)
    
    bench = jax.jit(bench_wrapper, static_argnums=(0, 1, 2, 3), donate_argnums=(4, 5, 6, 7))

    for _ in range(iterations):
        model_state, rngs_state, action_buffer, value_buffer = bench(model_def, rngs_def, batch_size, context_size, model_state, rngs_state, action_buffer, value_buffer)

    action_buffer.block_until_ready()
    value_buffer.block_until_ready()
    print("Warmup finished.")

    print("Benchmarking...")
    start_time = time.time()

    for _ in range(iterations):
        model_state, rngs_state, action_buffer, value_buffer = bench(model_def, rngs_def, batch_size, context_size, model_state, rngs_state, action_buffer, value_buffer)

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

    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        model_state, rngs_state, action_buffer, value_buffer = bench(model_def, rngs_def, batch_size, context_size, model_state, rngs_state, action_buffer, value_buffer)
        action_buffer.block_until_ready()
        value_buffer.block_until_ready()


def bench_wrapper(model_def, rngs_def, batch_size: int, context_size: int, model_state, rngs_state, action_buffer, value_buffer):
    model: TransformerActorCritic = nnx.merge(model_def, model_state)
    rngs: nnx.Rngs = nnx.merge(rngs_def, rngs_state)

    action_buffer, value_buffer = bench_fn(model, rngs, batch_size, context_size, action_buffer, value_buffer)

    model_def, model_state = nnx.split(model)
    rngs_def, rngs_state = nnx.split(rngs)

    return model_state, rngs_state, action_buffer, value_buffer


def bench_fn(model: TransformerActorCritic, rngs: nnx.Rngs, batch_size: int, context_size: int, action_buffer, value_buffer):
    def body(i, input: tuple[TransformerActorCritic, nnx.Rngs, Array, Array]):
        model, rngs, ab, vb = input

        obs = Observation(
            agents_view=random.normal(
                rngs.default(), (batch_size, 1, 10), dtype=jnp.bfloat16
            ),
            time_steps=jnp.full((1, 1), i),
            last_action=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            last_reward=jnp.zeros((batch_size, 1), dtype=jnp.bfloat16),
            action_mask=None,
        )
        value, policy = model(obs, True)
        action = policy.sample(seed=rngs.default())
        
        # Ensure action is the right type before squeezing
        action_array = jnp.asarray(action)
        value_array = jnp.asarray(value)

        ab = ab.at[:, i].set(jnp.squeeze(action_array))
        vb = vb.at[:, i].set(jnp.squeeze(value_array))

        return model, rngs, ab, vb

    model, rngs, action_buffer, value_buffer = nnx.fori_loop(
        0, context_size, body, (model, rngs, action_buffer, value_buffer)
    )
    return action_buffer, value_buffer


if __name__ == "__main__":
    main()
