# import jumanji
# from jumanji import Environment

import time
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
    kv_cache = model.create_kv_cache(rollout.batch_size, rollout.trajectory_length)

    def _step(i, x):
        rollout, rngs, env_state, ts, kv_cache = x

        action_key = rngs.action()
        env_key = rngs.env()

        value, policy, kv_cache = model(add_seq_dim(ts), kv_cache)
        
        action = policy.sample(seed=action_key)
        log_prob = policy.log_prob(action).squeeze(-1)
        action = action.squeeze(-1)
        value = value.squeeze(-1)
        # jax.debug.breakpoint()

        env_state, next_timestep = env.step(env_state, action, env_key)
        # jax.debug.breakpoint()

        rollout.store(
            step=i,
            obs=ts.obs,
            action=next_timestep.last_action,
            reward=next_timestep.last_reward,
            log_prob=log_prob,
            value=value
        )

        return rollout, rngs, env_state, next_timestep, kv_cache
    
    rollout, rngs, _, _, _ = nnx.fori_loop(
        0,
        rollout.trajectory_length,
        _step,
        init_val=(rollout, rngs, env_state, timestep, kv_cache)
    )

    rollout.calculate_advantage(discount=0.99, gae_lambda=0.95)

    return rollout, env_state, rngs


def main():
    batch_size = 4096
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
                ffn_size=512,
                gtrxl_gate=True,
                gtrxl_bias=2.0,
                glu=False,
                max_seq_length=length,
            ),
            hidden_features=128,
            num_layers=8,
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
    jitted_evaluate = nnx.jit(evaluate, static_argnums=(3,))
    for _ in range(10):
        rollout, _, _ = jitted_evaluate(model, rollout, rngs, env)
        rollout.rewards.block_until_ready()
        rollout.actions.block_until_ready()
        rollout.log_prob.block_until_ready()
        rollout.values.block_until_ready()
    
    start_time = time.time()
    rollout, _, _ = jitted_evaluate(model, rollout, rngs, env)
    end_time = time.time()
    rollout.rewards.block_until_ready()
    rollout.actions.block_until_ready()
    rollout.log_prob.block_until_ready()
    rollout.values.block_until_ready()
    print(f"Time taken: {end_time - start_time} seconds")

    total_steps = rollout.trajectory_length * batch_size
    print(f"Steps per second: {total_steps / (end_time - start_time)}")

    print(rollout.actions)
    # pass


if __name__ == '__main__':
    main()
