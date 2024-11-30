from jax import numpy as jnp
from flax import nnx

from jaxrl.experiment import Experiment
from jaxrl.envs.wrapper import EnvWrapper
from jaxrl.envs.envpool import EnvPoolWrapper
from jaxrl.model import create_learner, create_mlp_model


class Trainer:
    pass


def train(experiment: Experiment):
    print("Training...")
    print(experiment.config)

    # return

    environment: EnvWrapper = EnvPoolWrapper(
        experiment.config.environment.name,
        experiment.config.environment.num_envs,
        experiment.environments_seed,
    )

    print(environment.action_spec)
    print(environment.observation_spec)

    state, time_step = environment.reset()
    # state, next_time_step = environment.step(state, jnp.zeros(environment.num_envs, dtype=jnp.int32))
    # # print(next_time_step)

    rngs = nnx.Rngs(params=experiment.params_seed, action=experiment.actions_seed)

    learner = create_learner(
        experiment.config.learner,
        environment.num_envs,
        environment.observation_spec,
        environment.action_spec,
        rngs=rngs,
    )

    actions = learner.act(time_step.observation, rngs)
    print(actions)
    # print(learner)
