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
    return [TimeStep(
        obs=ts.obs[i],
        time=ts.time[i],
        terminated=ts.terminated[i],
        last_action=ts.last_action[i],
        last_reward=ts.last_reward[i],
        action_mask=None, #ts.action_mask[i],
    ) for i in range(batch_size)]


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

def evaluate(
    run_token: str,
    selector: Optional[str],
    seed: int,
    steps: Optional[int],
    rounds: int,
    verbose: bool,
):
    experiment = Experiment.load(run_token, base_dir="results")

    max_steps = steps or experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps, selector=selector)
    rngs = nnx.Rngs(default=seed)

    models = load_policy(experiment, env, max_steps, rngs)
    if len(models) < 2:
        raise typer.BadParameter("Need at least two checkpoints to evaluate head-to-head.")

    team_size = env.num_agents // 2

    for _ in progress.track(range(rounds), description="Ranking rounds"):
        m = random.choices(models, k=2)
        lineup = [m[0]] * team_size + [m[1]] * team_size
        policies = [JittedPolicy(p.model) for p in lineup]

        out = _round(env, policies, max_steps, rngs)
        ranking = np.argsort(out)
        winner = m[ranking[1]].rating
        loser = m[ranking[0]].rating

        winner, loser = trueskill.rate_1vs1(winner, loser)
        m[ranking[1]].rating = winner
        m[ranking[0]].rating = loser

        if verbose:
            console.print(f"{m[0].step} vs {m[1].step}")
            console.print(f"  {m[0].step}: {format_skill(m[0].rating)}")
            console.print(f"  {m[1].step}: {format_skill(m[1].rating)}")

    # Final summary sorted by rating.mu
    models_sorted = sorted(models, key=lambda r: r.rating.mu, reverse=True)
    console.print("Final ratings:")
    for rec in models_sorted:
        console.print(f"- step {rec.step}: {format_skill(rec.rating)}")


@app.command()
def main(
    run: str = typer.Option(
        ..., help="Existing experiment run token (under results/)", rich_help_panel="Input"
    ),
    selector: Optional[str] = typer.Option(
        None, help="Select a specific env when using a multi env config."
    ),
    seed: int = typer.Option(0, help="Random seed for RNGs."),
    steps: Optional[int] = typer.Option(
        None, help="Override steps per episode (defaults to config)."
    ),
    rounds: int = typer.Option(1000, help="Number of head-to-head rounds to run."),
    verbose: bool = typer.Option(False, help="Print per-round rating updates."),
):
    evaluate(run, selector, seed, steps, rounds, verbose)


if __name__ == "__main__":
    app()
