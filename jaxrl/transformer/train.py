# import jumanji
# from jumanji import Environment

import time
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax

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
    kv_cache = model.create_kv_cache(rollout.batch_size, rollout.trajectory_length, dtype=jnp.bfloat16)

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

    return rollout, rngs

def ppo_loss(model: TransformerActorCritic, rollout: Rollout, batch_idx: jax.Array, rngs: nnx.Rngs):
    vf_coef = 0.5
    entropy_coef = 0.01
    vf_clip = 0.2
    
    batch_obs = rollout.obs.value[batch_idx]
    batch_target = rollout.targets.value[batch_idx]
    batch_log_prob = rollout.log_prob.value[batch_idx]
    batch_actions = rollout.actions.value[batch_idx]
    batch_advantage = rollout.advantages.value[batch_idx]
    batch_values = rollout.values.value[batch_idx, :-1]
    batch_rewards = rollout.rewards.value[batch_idx]

    # TODO: make this conditional
    batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]
    print(batch_obs.shape)
    print(positions.shape)

    values, policy, _ = model(TimeStep(
        obs=batch_obs,
        time=positions,
        last_action=batch_actions,
        last_reward=batch_rewards,
        step_type=jnp.zeros_like(batch_target, dtype=jnp.int32),
        action_mask=None
    ))
    log_probs = policy.log_prob(batch_actions)

    value_pred_clipped = batch_values + jnp.clip(values - batch_values, -vf_clip, vf_clip)

    value_losses = jnp.square(values - batch_target)
    value_losses_clipped = jnp.square(value_pred_clipped - batch_target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_probs - batch_log_prob)

    loss_actor1 = ratio * batch_advantage
    loss_actor2 = jnp.clip(ratio, 1.0 - vf_clip, 1.0 + vf_clip) * batch_advantage

    actor_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()

    # Entropy regularization
    entropy = policy.entropy()
    entropy_loss = -entropy.mean()

    total_loss = vf_coef * value_loss + actor_loss + entropy_coef * entropy_loss

    return total_loss



def train(optimizer: nnx.Optimizer, rollout: Rollout, rngs: nnx.Rngs, env: Environment):
    rollout, rngs = evaluate(optimizer.model, rollout, rngs, env)
    
    minibatch_count = 8
    minibatch_size = rollout.batch_size // minibatch_count

    batch_idx = jnp.arange(rollout.batch_size, dtype=jnp.int32)
    batch_idx = jnp.reshape(batch_idx, (minibatch_count, minibatch_size))

    def _step(i, x):
        optimizer, rollout, rngs = x
        grads = nnx.grad(ppo_loss)(optimizer.model, rollout, batch_idx[i], rngs)
        optimizer.update(grads)
        return optimizer, rollout, rngs

    optimizer, rollout, rngs = nnx.fori_loop(0, minibatch_count, _step, init_val=(optimizer, rollout, rngs))

    return optimizer, rngs



def main():
    batch_size = 4096
    length = 128

    env = NBackMemory(n=2, max_value=5, length=length)
    env = VmapWrapper(env, batch_size)
    obs_spec = env.observation_spec
    action_spec = env.action_spec

    rngs = nnx.Rngs(default=42)
    rollout = Rollout(batch_size, length, obs_spec)
# num_layers: 12
# num_heads: 12
# d_model: 768
# ffn_size: 2048
    model = TransformerActorCritic(
        TransformerActorCriticConfig(
            obs_encoder=LinearObsEncoderConfig(),
            transformer_block=TransformerBlockConfig(
                num_heads=4,
                ffn_size=512,
                gtrxl_gate=False,
                gtrxl_bias=2.0,
                glu=False,
                max_seq_length=length,
            ),
            hidden_features=128,
            num_layers=8,
            activation='gelu',
            norm='layer_norm',
            kernel_init='glorot_uniform',
            dtype='bfloat16',
            param_dtype='float32'
        ),
        obs_spec.shape[0],
        action_spec.num_actions,
        rngs=rngs
    )

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-4))

    
    jitted_train = nnx.jit(train, static_argnums=(3,))
    for _ in range(10):
        optimizer, rngs = jitted_train(optimizer, rollout, rngs, env)
        rollout.rewards.block_until_ready()
        model.value_head.kernel.value.block_until_ready()
    
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        start_time = time.time()
        optimizer, rngs = jitted_train(optimizer, rollout, rngs, env)
        rollout.rewards.block_until_ready()
        model.value_head.kernel.value.block_until_ready()
        end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    total_steps = rollout.trajectory_length * batch_size
    print(f"Steps per second: {total_steps / (end_time - start_time)}")

    print(rollout.actions)
    # pass


if __name__ == '__main__':
    main()
