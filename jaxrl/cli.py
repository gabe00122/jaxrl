import os
import typer
from flax import nnx
import jax
from jax import numpy as jnp

from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.create import create_client, create_env
from jaxrl.experiment import Experiment
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.train import add_seq_dim, train_run
import shutil


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def enjoy(name: str, base_dir: str = "results", seed: int = 0):
    experiment: Experiment = Experiment.load(name, base_dir)
    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps)
    rngs = nnx.Rngs(default=seed)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )

    # optimizer = nnx.ModelAndOptimizer(
    #     model=model,
    #     tx=create_optimizer(
    #         experiment.config.learner.optimizer,
    #         experiment.config.update_steps
    #         * experiment.config.learner.trainer.minibatch_count
    #         * experiment.config.learner.trainer.epoch_count
    #     ),
    #     wrt=nnx.Param
    # )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        model = checkpointer.restore_latest(model)

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

    for _ in range(3):
        kv_cache = model.initialize_carry(env.num_agents, rngs=rngs)

        env_state, timestep = env.reset(rngs.env())
        client.render(env_state, timestep)
        for _ in range(max_steps):
            env_state, timestep, kv_cache, rngs = step(
                timestep, kv_cache, env_state, rngs
            )
            # timestep = timestep._replace(last_action=timestep.last_action.at[0].set(1))
            client.render(env_state, timestep)

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
