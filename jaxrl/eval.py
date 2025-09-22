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
from jaxrl.train import add_seq_dim
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


@partial(nnx.jit, static_argnums=(0, 3))
def _round(env: Environment, policy1: TransformerActorCritic, policy2: TransformerActorCritic, max_steps: int, rngs: nnx.Rngs):
    teams = env.teams
    team_size = env.num_agents // 2

    policy1_carry = policy1.initialize_carry(team_size, rngs)
    policy2_carry = policy2.initialize_carry(team_size, rngs)

    policy1_idx = jnp.where(teams == 0, fill_value=-1, size=team_size)
    policy2_idx = jnp.where(teams == 1, fill_value=-1, size=team_size)

    state, ts = env.reset(rngs.env())

    def _step(_, x):
        state, ts, policy1_carry, policy2_carry, policy1_reward, policy2_reward, rngs = x

        ts = add_seq_dim(ts)

        policy1_ts = jax.tree.map(lambda item: item[policy1_idx], ts)
        policy2_ts = jax.tree.map(lambda item: item[policy2_idx], ts)

        _, a1, policy1_carry = policy1(policy1_ts, policy1_carry)
        _, a2, policy2_carry = policy2(policy2_ts, policy2_carry)

        a1 = a1.sample(seed=rngs.action()).squeeze(-1)
        a2 = a2.sample(seed=rngs.action()).squeeze(-1)

        actions = jnp.zeros((env.num_agents,), jnp.int32)
        actions = actions.at[policy1_idx].set(a1)
        actions = actions.at[policy2_idx].set(a2)

        state, ts = env.step(state, actions, rngs.env())

        policy1_reward = policy1_reward + policy1_ts.last_reward.sum()
        policy2_reward = policy2_reward + policy2_ts.last_reward.sum()

        return state, ts, policy1_carry, policy2_carry, policy1_reward, policy2_reward, rngs

    _, _, _, _, policy1_reward, policy2_reward, rngs = nnx.fori_loop(
        0,
        max_steps,
        _step,
        (state, ts, policy1_carry, policy2_carry, jnp.float32(0.0), jnp.float32(0.0), rngs)
    )

    return policy1_reward, policy2_reward, rngs


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
            policy = PolicyRecord(experiment.unique_token, step, trueskill.Rating(), model)
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

    env, task_count = create_env(experiment.config.environment, max_steps, vec_count=16, env_name=env_name)
    rngs = nnx.Rngs(default=42)

    league: list[PolicyRecord] = [] 

    for name in run_tokens:
        console.print(f"Loading: {name}")
        experiment = Experiment.load(name, base_dir="results")
        policies = load_policy(experiment, env, max_steps, task_count, rngs)
        league.extend(policies)


    plot = LiveRankingsPlot()

    for i in progress.track(range(rounds), description="Ranking rounds", console=console):
        agents = random.choices(league, k=2)
        policies = [p.model for p in agents]

        console.print(f"Round: {i}")
        policy1_reward, policy2_reward, rngs = _round(env, policies[0], policies[1], max_steps, rngs)
        winner, loser = agents if policy1_reward > policy2_reward else (agents[1], agents[0])

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
