# import jumanji
# from jumanji import Environment

import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange

from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.vmap_wrapper import VmapWrapper
from jaxrl.transformer.network import LinearObsEncoderConfig, TransformerActorCritic, TransformerActorCriticConfig, TransformerBlockConfig
from jaxrl.transformer.rollout import ObservationSpec, Rollout
from jaxrl.types import TimeStep

def add_seq_dim(ts: TimeStep):
    return TimeStep(
        obs=rearrange(ts.obs, 'b ... -> b 1 ...'),
        time=rearrange(ts.time, 'b ... -> b 1 ...'),
        last_action=rearrange(ts.last_action, 'b ... -> b 1 ...'),
        last_reward=rearrange(ts.last_reward, 'b ... -> b 1 ...'),
        step_type=rearrange(ts.step_type, 'b ... -> b 1 ...'),
        action_mask=rearrange(ts.action_mask, 'b ... -> b 1 ...') if ts.action_mask is not None else None,
    )

def evaluate(model: TransformerActorCritic, rollout: Rollout, rngs: nnx.Rngs, env: Environment):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)

    def _step(i, x):
        rollout, rngs, env_state, ts = x

        action_key = rngs.action()
        env_key = rngs.env()

        value, policy = model(add_seq_dim(ts), True)
        action = policy.sample(action_key)

        env_state, next_timestep = env.step(env_state, action, env_key)

        rollout.store(
            step=ts.time,
            obs=ts.obs,
            action=next_timestep.last_action,
            reward=next_timestep.last_reward,
            log_prob=policy.log_prob(action),
            value=value
        )

        return rollout, rngs, env_state, next_timestep
    
    rollout, rngs, _, _ = nnx.fori_loop(
        0,
        rollout.batch_size,
        _step,
        init_val=(rollout, rngs, env_state, timestep)
    )

    return rollout, env_state, rngs


def main():
    batch_size = 32
    length = 128

    env = NBackMemory(n=2, max_value=5, length=length)
    env = VmapWrapper(env, batch_size)
    obs_spec = env.observation_spec
    action_spec = env.action_spec

    rngs = nnx.Rngs(default=42)
    rollout = Rollout(batch_size, length, obs_spec)

    model = TransformerActorCritic(
        TransformerActorCriticConfig(
            obs_encoder=LinearObsEncoderConfig(),
            transformer_block=TransformerBlockConfig(
                num_heads=4,
                ffn_size=128,
                gtrxl_gate=True,
                gtrxl_bias=2.0,
                glu=False,
                max_seq_length=length,
            ),
            hidden_features=128,
            num_layers=3,
            activation='gelu',
            norm='layer_norm',
            kernel_init='glorot_uniform',
            dtype='float32',
            param_dtype='float32'
        ),
        obs_spec.shape[0],
        action_spec.num_actions,
        rngs=rngs
    )
    model.create_kv_cache(batch_size, length)

    rollout, _, _ = evaluate(model, rollout, rngs, env)
    print(rollout.actions)
    # pass


if __name__ == '__main__':
    main()
