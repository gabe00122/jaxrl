from jax import numpy as jnp
from flax import nnx

from jaxrl.checkpointer import Checkpointer
from jaxrl.experiment import Experiment
from jaxrl.envs.wrapper import EnvWrapper, StepType, TimeStep
from jaxrl.envs.envpool import EnvPoolWrapper
from jaxrl.logger import JaxLogger
from jaxrl.model import create_learner
from jaxrl.systems.actor_critic import ActorCriticLearner
from jaxrl.systems.types import Transition
from jaxrl.types import Action, Observation


@nnx.jit
def act(learner: ActorCriticLearner, observation: Observation, rngs: nnx.Rngs) -> Action:
    return learner.act(observation, rngs)

@nnx.jit
def learn(learner: ActorCriticLearner, transition: Transition, rngs: nnx.Rngs):
    learner.learn(transition, rngs)


def train(experiment: Experiment):
    print("Training...")
    print(experiment.config)

    environment: EnvWrapper = EnvPoolWrapper(
        experiment.config.environment.name,
        experiment.config.environment.num_envs,
        experiment.environments_seed,
    )

    logger = JaxLogger(experiment.config, experiment.unique_token)
    checkpointer = Checkpointer(experiment.checkpoints_dir)

    print(environment.action_spec)
    print(environment.observation_spec)

    rngs = nnx.Rngs(
        default=experiment.default_seed,
        params=experiment.params_seed,
        action=experiment.actions_seed,
    )

    learner = create_learner(
        experiment.config.learner,
        environment.num_envs,
        environment.observation_spec,
        environment.action_spec,
        rngs=rngs,
    )

    state, time_step = environment.reset()

    for global_step in range(experiment.config.environment.max_steps):
        action = act(learner, time_step.observation, rngs)
        state, next_time_step = environment.step(state, action)

        transition = make_transition(time_step, action, next_time_step)

        learn(learner, transition, rngs)

        time_step = next_time_step

        if global_step % experiment.config.logger.log_rate == experiment.config.logger.log_rate - 1:
            logger.log(learner.metrics.compute(), global_step)
            learner.metrics.reset()

            checkpointer.save(learner, global_step)

    checkpointer.save(learner, experiment.config.environment.max_steps)
    checkpointer.close()
    logger.close()


def make_transition(time_step: TimeStep, action: Action, next_time_step: TimeStep) -> Transition:
    return Transition(
        observation=time_step.observation,
        action=action,
        reward=next_time_step.reward,
        next_observation=next_time_step.observation,
        terminated=next_time_step.step_type == StepType.LAST,
        truncated=jnp.bool(False),
    ) 
