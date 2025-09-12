from dataclasses import dataclass
import random
from typing import Any
from einops import rearrange
import jax
import numpy as np
from jax import numpy as jnp
from flax import nnx

import trueskill

from functools import partial

from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.env_config import create_env
from jaxrl.envs.environment import Environment
from jaxrl.experiment import Experiment
from jaxrl.model.network import TransformerActorCritic
from jaxrl.types import TimeStep


@dataclass
class PolicyRecord:
    name: str
    step: int
    rating: trueskill.Rating
    model: Any


@partial(nnx.jit, static_argnums=1)
def create_carry(model, num_agents: int, rngs):
    return model.initialize_carry(num_agents, rngs=rngs)


@partial(nnx.jit, donate_argnums=(2,))
def sample_actions(model, timestep, carry, rngs):
    action_key = rngs.action()
    _, policy, carry = model(timestep, carry)
    actions = policy.sample(seed=action_key)
    actions = actions.squeeze()

    return actions, carry


@partial(jax.jit, static_argnums=(0,))
def env_reset(env: Environment, rng_key):
    env_state, timestep = env.reset(rng_key)
    return env_state, split_timestep(timestep)


@partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
def env_step(env: Environment, env_state, actions, rng_key):
    env_state, timestep = env.step(env_state, actions, rng_key)
    return env_state, split_timestep(timestep)


def split_timestep(ts: TimeStep):
    ts = jax.tree.map(lambda x: rearrange(x, "b ... -> b 1 1 ..."), ts)
    batch_size = ts.obs.shape[0]
    return [TimeStep(
        obs=ts.obs[i],
        time=ts.time[i],
        terminated=ts.terminated[i],
        last_action=ts.last_action[i],
        last_reward=ts.last_reward[i],
        action_mask=None, #ts.action_mask[i],
    ) for i in range(batch_size)]


def _round(env: Environment, policies: list, max_steps: int, rngs: nnx.Rngs):
    # num_agents = env.num_agents

    teams = np.asarray(env.teams)
    unique_teams = np.unique(teams)

    team_rewards = np.zeros_like(unique_teams, dtype=np.float64)

    carries = [create_carry(policy, 1, rngs) for policy in policies]
    env_state, timestep = env_reset(env, rngs.env())

    np_actions = np.zeros((len(policies,)), jnp.int32)
    
    for _ in range(max_steps):
        for i, policy in enumerate(policies):
            action, carry = sample_actions(policy, timestep[i], carries[i], rngs)
            carries[i] = carry
            np_actions[i] = action
        
        actions = jnp.asarray(np_actions)
        
        env_state, timestep = env_step(env, env_state, actions, rngs.env())

        for i, ts in enumerate(timestep):
            team_idx = teams[i]
            team_rewards[team_idx] += ts.last_reward.item()
    
    return team_rewards


def rank_policies(env: Environment, policies: list):
    pass

def load_policy(experiment: Experiment, env, max_steps, rngs: nnx.Rngs):
    model_template = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )

    policies = []

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        for step in checkpointer.mngr.all_steps():
            model = checkpointer.restore(model_template, step)
            policy = PolicyRecord(experiment.unique_token, step, trueskill.Rating(), model)
            policies.append(policy)

    return policies

def format_skill(rating: trueskill.Rating):
    return f"mu: {rating.mu}, sigma: {rating.sigma}"

def main():
    experiment = Experiment.load("loyal-bear-3v0443")
    print(experiment)

    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps, selector="koth")
    rngs = nnx.Rngs(default=42)

    models = load_policy(experiment, env, max_steps, rngs)

    for i in range(50):
        m = random.choices(models, k=2)
        lineup = [m[0]] * 4 + [m[1]] * 4
        p = [p.model for p in lineup]

        out = _round(env, p, max_steps, rngs)
        ranking = np.argsort(out)
        print(f"{m[0].step} vs {m[1].step}")
        winner = m[ranking[1]].rating
        losser = m[ranking[0]].rating

        winner, losser = trueskill.rate_1vs1(winner, losser)

        m[ranking[1]].rating = winner
        m[ranking[0]].rating = losser

        print(m[0].rating)
        print(m[1].rating)
    
    for m in models:
        print(f"{m.step} - {m.rating}")



if __name__ == '__main__':
    main()
