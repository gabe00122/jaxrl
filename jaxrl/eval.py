import csv
from dataclasses import dataclass
import random
from typing import Any, Optional
from einops import rearrange
import jax
import numpy as np
from jax import numpy as jnp
from flax import nnx

import trueskill

from functools import partial

import typer
from rich import progress
from rich.console import Console

from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.env_config import create_env
from jaxrl.envs.environment import Environment
from jaxrl.experiment import Experiment
from jaxrl.model.network import TransformerActorCritic
from jaxrl.types import TimeStep
from jaxrl.utils.live_skill_plot import LiveRankingsPlot


console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)


@dataclass
class PolicyRecord:
    name: str
    step: int
    rating: trueskill.Rating
    model: Any

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
    return [TimeStep(**{key: value[i] if value is not None else None for key, value in ts._asdict().items()}) for i in range(batch_size)]


@partial(jax.jit, static_argnums=(0, 2))
def _initialize_carry(graphdef, state, num_agents: int, rng_key):
    model = nnx.merge(graphdef, state)
    return model.initialize_carry(num_agents, rngs=nnx.Rngs(rng_key))


@partial(jax.jit, static_argnums=(0,), donate_argnums=(3,))
def _sample(graphdef, state, timestep, carry, _key):
    model = nnx.merge(graphdef, state)
    _, policy, carry = model(timestep, carry)
    action = policy.sample(seed=_key)
    action = action.squeeze()

    return action, carry


class JittedPolicy:
    """
    Pre split the policy to avoid the nnx overhead
    """
    def __init__(self, model) -> None:
        self._graphdef, self._state = nnx.split(model)

    def sample_action(self, ts: TimeStep, carry, rng_key: jax.Array):
        return _sample(self._graphdef, self._state, ts, carry, rng_key)
    
    def initialize_carry(self, num_agents: int, rng_key):
        return _initialize_carry(self._graphdef, self._state, num_agents, rng_key)


def _round(env: Environment, policies: list, max_steps: int, rngs: nnx.Rngs):
    teams = np.asarray(env.teams)
    unique_teams = np.unique(teams)

    team_rewards = np.zeros_like(unique_teams, dtype=np.float64)

    carries = [policy.initialize_carry(1, rngs.carry()) for policy in policies]
    env_state, timestep = env_reset(env, rngs.env())

    np_actions = np.zeros((len(policies,)), np.int32)
    
    for _ in range(max_steps):
        for i, policy in enumerate(policies):
            action, carry = policy.sample_action(timestep[i], carries[i], rngs.action())
            carries[i] = carry
            np_actions[i] = action
        
        actions = jnp.asarray(np_actions)
        
        env_state, timestep = env_step(env, env_state, actions, rngs.env())

        for i, ts in enumerate(timestep):
            team_idx = teams[i]
            team_rewards[team_idx] += ts.last_reward.item()
    
    return team_rewards


def load_policy(experiment: Experiment, env, max_steps: int, task_count: int, rngs: nnx.Rngs) -> list[PolicyRecord]:
    model_template = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        task_count=task_count,
        rngs=rngs,
    )

    policies = []

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        for step in checkpointer.mngr.all_steps():
            model = checkpointer.restore(model_template, step)
            policy = PolicyRecord(experiment.unique_token, step, trueskill.Rating(), JittedPolicy(model))
            policies.append(policy)

    return policies

def format_skill(rating: trueskill.Rating):
    return f"mu: {rating.mu}, sigma: {rating.sigma}"

def evaluate(
    run_tokens: list[str],
    env_name: Optional[str],
    seed: int,
    rounds: int,
):
    console = Console()
    
    # assume all the runs have the same environment config, task id's will differ if the environment used are different
    # one work around is to keep the env you want to test the same among all runs and the first one defined in the config
    experiment = Experiment.load(run_tokens[0], base_dir="results")
    max_steps = experiment.config.max_env_steps

    env, task_count = create_env(experiment.config.environment, max_steps, env_name=env_name)
    rngs = nnx.Rngs(default=seed)

    league: list[PolicyRecord] = [] 

    for name in run_tokens:
        console.print(f"Loading: {name}")
        experiment = Experiment.load(name, base_dir="results")
        policies = load_policy(experiment, env, max_steps, task_count, rngs)
        league.extend(policies)

    team_size = env.num_agents // 2

    plot = LiveRankingsPlot()

    for i in progress.track(range(rounds), description="Ranking rounds", console=console):
        agents = random.choices(league, k=2)
        lineup = [agents[0]] * team_size + [agents[1]] * team_size
        policies = [p.model for p in lineup]

        console.print(f"Round: {i}")
        out = _round(env, policies, max_steps, rngs)
        ranking = np.argsort(out)
        winner = agents[ranking[1]]
        loser = agents[ranking[0]]

        winner_rating, loser_rating = trueskill.rate_1vs1(winner.rating, loser.rating)
        winner.rating = winner_rating
        loser.rating = loser_rating

        plot.update(league)

    with open("rankings.csv", "w", newline='') as f:
        rankings_writer = csv.writer(f)

        for policy in league:
            rankings_writer.writerow([policy.name, policy.step, policy.rating.mu, policy.rating.sigma])


@app.command()
def main(
    runs: list[str] = typer.Option(
        ..., help="Existing experiment run token (under results/)", rich_help_panel="Input"
    ),
    env: Optional[str] = typer.Option(
        None, help="Select a specific env when using a multi env config."
    ),
    seed: int = typer.Option(0, help="Random seed for RNGs."),
    rounds: int = typer.Option(1000, help="Number of head-to-head rounds to run."),
):
    evaluate(runs, env_name=env, seed=seed, rounds=rounds)


if __name__ == "__main__":
    app()
