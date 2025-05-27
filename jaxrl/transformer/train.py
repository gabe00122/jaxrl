import jumanji
from jumanji import Environment

import jax
import jax.numpy as jnp
from flax import nnx

from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.rollout import ObservationSpec, Rollout


def evaluate(model: TransformerActorCritic, rollout: Rollout, rngs: nnx.Rngs, env: Environment):
    reset_keys = jax.random.split(rngs.env(), rollout.batch_size)
    env_state, timestep = jax.vmap(env.reset)(reset_keys)
    timestep.

    def _step(rollout, env_state, rngs):
        # rollout, env_state, rngs = x
        action = jnp.array(0, dtype=jnp.int32)
        env_state = env.step(env_state, action)

        return rollout, env_state, rngs
    
    step_keys = jax.random.split(rngs.env(), (rollout.batch_size, rollout.trajectory_length))
    rollout, env_state, rngs = nnx.scan(_step, in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0, 0), length=rollout.batch_size)(rollout, env_state, step_keys)

    return rollout, env_state, rngs


def main():
    env = jumanji.make('Snake-v1')
    obs_spec = ObservationSpec(env.observation_spec().grid.shape)

    rngs = nnx.Rngs(default=42)
    rollout = Rollout(3, 32, obs_spec)
    rollout, _, _ = evaluate(None, rollout, rngs, env)
    print(rollout)
    # pass


if __name__ == '__main__':
    main()
