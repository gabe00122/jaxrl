"""Export JAX checkpoint weights to .npz + reference output for TF.js PoC.

Usage:
    uv run python tfjs_poc/export_weights.py hungry-hawk-ommxca
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from mapox import EnvironmentFactory, TimeStep

from jaxrl.checkpointer import Checkpointer
from jaxrl.experiment import Experiment
from jaxrl.model.network import TransformerActorCritic
from jaxrl.util import add_seq_dim


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python tfjs_poc/export_weights.py <experiment_name>")
        sys.exit(1)

    name = sys.argv[1]
    out_dir = Path("tfjs_poc/weights")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment and environment
    experiment = Experiment.load(name, "results")
    config = experiment.config

    env_factory = EnvironmentFactory()
    env, task_count = env_factory.create_env(
        config.environment, config.max_env_steps, 1, None
    )

    print(f"obs_spec: {env.observation_spec}")
    print(f"action_spec: n={env.action_spec.n}")
    print(f"num_agents: {env.num_agents}")
    print(f"task_count: {task_count}")

    # Build model and load checkpoint
    rngs = nnx.Rngs(42)
    model = TransformerActorCritic(
        config.learner.model,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=config.max_env_steps,
        task_count=task_count,
        rngs=rngs,
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        model = checkpointer.restore_latest(model)

    print("Checkpoint loaded.")

    # ── Extract weights ──
    state = nnx.state(model, nnx.Param)
    flat = state.flat_state()

    weights = {}
    for path_tuple, variable in flat:
        key = ".".join(str(p) for p in path_tuple)
        arr = np.array(variable.raw_value, dtype=np.float32)
        weights[key] = arr
        print(f"  {key}: {arr.shape}")

    np.savez(out_dir / "model.npz", **weights)
    print(f"Saved {len(weights)} weight arrays to {out_dir / 'model.npz'}")

    # ── Save model config ──
    obs_spec = env.observation_spec
    model_config = {
        "hidden_features": config.learner.model.hidden_features,
        "num_layers": config.learner.model.num_layers,
        "num_heads": config.learner.model.layer.history.num_heads,
        "num_kv_heads": config.learner.model.layer.history.num_kv_heads,
        "head_dim": config.learner.model.layer.history.head_dim,
        "rope_max_wavelength": config.learner.model.layer.history.rope_max_wavelength,
        "ffn_size": config.learner.model.layer.feed_forward.size,
        "action_dim": env.action_spec.n,
        "max_seq_length": config.max_env_steps,
        "obs_shape": list(obs_spec.shape),
        "obs_max_value": list(obs_spec.max_value)
        if isinstance(obs_spec.max_value, tuple)
        else obs_spec.max_value,
        "obs_one_hot_total": sum(obs_spec.max_value)
        if isinstance(obs_spec.max_value, tuple)
        else int(obs_spec.max_value),
        "conv_kernels": [list(k) for k in config.learner.model.obs_encoder.kernels],
        "conv_strides": [list(s) for s in config.learner.model.obs_encoder.strides],
        "conv_channels": list(config.learner.model.obs_encoder.channels),
        "value_hidden_dim": config.learner.model.value_hidden_dim,
        "value_n_logits": config.learner.model.value.n_logits,
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved config to {out_dir / 'config.json'}")

    # ── Reference forward pass (float32 to match TF.js) ──
    # Rebuild model with float32 compute dtype so the reference matches JS output.
    f32_config = config.learner.model.model_copy(update={"dtype": "float32"})
    f32_model = TransformerActorCritic(
        f32_config,
        env.observation_spec,
        env.action_spec.n,
        max_seq_length=config.max_env_steps,
        task_count=task_count,
        rngs=nnx.Rngs(42),
    )
    # Copy the loaded weights (already float32) into the new model
    f32_state = nnx.state(model, nnx.Param)
    nnx.update(f32_model, f32_state)

    # cuDNN attention doesn't support float32 — switch to XLA
    for layer in f32_model.layers:
        layer.history.attention_impl = "xla"

    num_agents = 1
    rng = jax.random.PRNGKey(123)

    # Create a deterministic dummy observation
    rng, obs_rng = jax.random.split(rng)
    dummy_obs = jax.random.randint(
        obs_rng,
        (num_agents,) + tuple(obs_spec.shape),
        0,
        # per-channel max for the last axis
        jnp.array(obs_spec.max_value)[None, None, :],
        dtype=jnp.int8,
    )
    dummy_reward = jnp.zeros((num_agents,), dtype=jnp.float32)
    dummy_last_action = jnp.zeros((num_agents,), dtype=jnp.int32)
    dummy_time = jnp.zeros((num_agents,), dtype=jnp.int32)

    dummy_ts = TimeStep(
        obs=dummy_obs,
        reward=dummy_reward,
        last_action=dummy_last_action,
        time=dummy_time,
        terminated=jnp.zeros((num_agents,), dtype=jnp.bool_),
        action_mask=None,
        task_ids=None,
    )

    # Initialize KV caches and run single-step inference (float32)
    carry = f32_model.initialize_carry(num_agents, rngs=nnx.Rngs(0))
    value_rep, policy, new_carry = f32_model(add_seq_dim(dummy_ts), carry)

    action_logits = policy.logits.squeeze(axis=1)  # remove seq dim
    action_probs = jax.nn.softmax(action_logits, axis=-1)

    reference = {
        "obs": np.array(dummy_obs, dtype=np.int8).tolist(),
        "reward": np.array(dummy_reward, dtype=np.float32).tolist(),
        "last_action": np.array(dummy_last_action, dtype=np.int32).tolist(),
        "time": np.array(dummy_time, dtype=np.int32).tolist(),
        "action_logits": np.array(action_logits, dtype=np.float32).tolist(),
        "action_probs": np.array(action_probs, dtype=np.float32).tolist(),
    }

    with open(out_dir / "reference.json", "w") as f:
        json.dump(reference, f)
    print(f"Saved reference output to {out_dir / 'reference.json'}")
    print(f"  action_probs: {np.array(action_probs[0])}")


if __name__ == "__main__":
    main()
