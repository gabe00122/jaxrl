import os
import typer
from flax import nnx
import jax
from jax import numpy as jnp

from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.create import create_env
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.return_2d import ReturnClient
from jaxrl.envs.memory.return_2d_colors import ReturnColorClient
from jaxrl.experiment import Experiment
from jaxrl.optimizer import create_optimizer
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.train import add_seq_dim, train_run
import shutil


app = typer.Typer(pretty_exceptions_show_locals=False)


def create_client(env: Environment):
    return ReturnColorClient(env)


@app.command()
def enjoy(name: str, base_dir: str = "results", seed: int = 0):
    experiment: Experiment = Experiment.load(name, base_dir)
    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps)

    obs_spec = env.observation_spec
    action_spec = env.action_spec
    rngs = nnx.Rngs(default=seed)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        obs_spec,
        action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(
        model=model,
        tx=create_optimizer(
            experiment.config.learner.optimizer, experiment.config.update_steps
        ),
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        optimizer = checkpointer.restore_latest(optimizer)

    model = optimizer.model

    kv_cache = model.create_kv_cache(env.num_agents)
    client = create_client(env)

    @nnx.jit
    def step(timestep, kv_cache, env_state, rngs):
        action_key = rngs.action()
        env_key = rngs.env()
        _, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        actions = policy.sample(seed=action_key)
        actions = jnp.squeeze(actions, axis=-1)

        env_state, timestep = env.step(env_state, actions, env_key)

        return env_state, timestep, kv_cache, rngs

    for _ in range(10):
        env_state, timestep = env.reset(rngs.env())
        client.render(env_state)
        for _ in range(max_steps):
            env_state, timestep, kv_cache, rngs = step(
                timestep, kv_cache, env_state, rngs
            )
            client.render(env_state)

    client.save_video()


@app.command("train")
def train_cmd(
    config: str = "./config/return.json",
    distributed: bool = False,
    base_dir: str = "./results",
):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(config, base_dir)

    train_run(experiment)


@app.command("profile")
def profile(config: str = "./config/return.json", distributed: bool = False, base_dir: str = "./results"):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(config, base_dir, create_directories=False)

    train_run(experiment, profile=True)


@app.command("clean")
def clean():
    subfolders = [f.path for f in os.scandir("./results") if f.is_dir()]

    for folder in subfolders:
        checkpoint_count = len([f.path for f in os.scandir(folder + '/checkpoints') if f.is_dir()])

        if checkpoint_count == 0:
            shutil.rmtree(folder)



if __name__ == "__main__":
    app()
