from jax import numpy as jnp
from flax import nnx

from jaxrl.experiment import Experiment
from jaxrl.envs.wrapper import EnvWrapper
from jaxrl.envs.envpool import EnvPoolWrapper
from jaxrl.model import create_mlp_model


class Trainer:
    pass


def train(experiment: Experiment):
    print("Training...")

    environment: EnvWrapper = EnvPoolWrapper(
        experiment.config.environment.name,
        experiment.config.environment.num_envs,
        experiment.environments_seed,
    )

    print(environment.action_spec)
    print(environment.observation_spec)

    # state, time_step = environment.reset()
    # state, next_time_step = environment.step(state, jnp.zeros(environment.num_envs, dtype=jnp.int32))
    # # print(next_time_step)

    rngs = nnx.Rngs(params=experiment.params_seed, action=experiment.actions_seed)

    model = create_mlp_model(
        experiment.config.learner.model,
        environment.observation_spec,
        environment.action_spec,
        rngs=rngs,
    )

    print(model)
