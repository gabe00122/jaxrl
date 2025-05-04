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
        glu = False,
        gtrxl_gate = False,
        activation="relu",
        kernel_init=nnx.initializers.orthogonal(0.01),
        rngs=rngs
    )
    transformer.create_kv_cache(144, 1024)

    print(transformer)


@nnx.jit
def bench(model: TransformerActorCritic, rngs: nnx.Rngs, batch_size: int, context_size: int):
    def body(i, input: tuple[TransformerActorCritic, nnx.Rngs, Array, Array]):
        model, rngs, ab, vb = input

        obs = Observation(
            agents_view=random.normal(rngs.default(), (batch_size, 2)),
            time_steps=jnp.full((1, 1), i),
            last_action=jnp.zeros(batch_size, dtype=jnp.int32),
            last_reward=jnp.zeros(batch_size, dtype=jnp.float32),
            action_mask=None,
        )
        value, policy = model(obs, True)
        action = policy.sample(seed=rngs.default())

        ab[:, i] = action
        vb[:, i] = value

        return model, rngs, ab, vb

    action_buffer = jnp.zeros((batch_size, context_size), dtype=jnp.int32)
    value_buffer = jnp.zeros((batch_size, context_size), dtype=jnp.float32)

    model, rngs, action_buffer, value_buffer = nnx.fori_loop(0, context_size, body, (model, rngs, action_buffer, value_buffer))
    return model


if __name__ == '__main__':
    main()
