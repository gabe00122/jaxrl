from calendar import c
from functools import partial
import stat
import chex
from jax import numpy as jnp
from flax import nnx
import jax
import numpy as np

from jaxrl.checkpointer import Checkpointer
from jaxrl.experiment import Experiment
from jaxrl.envs.wrapper import EnvWrapper, StepType, TimeStep
from jaxrl.envs.envpool import EnvPoolWrapper
from jaxrl.logger import JaxLogger
from jaxrl.model import create_learner
from jaxrl.systems.actor_critic import ActorCriticLearner
from jaxrl.systems.types import Transition
from jaxrl.types import Action, Observation
import jaxrl.utils.video_writter as vw


class Trainer:
    def __init__(self, learner: ActorCriticLearner, rngs: nnx.Rngs):
        learner_def, self.learner_state = nnx.split(learner)
        rngs_def, self.rngs_state = nnx.split(rngs)

        self.learner_def = learner_def
        self.rngs_def = rngs_def

        def _act(fn_learner_state, fn_rngs_state, observation):
            fn_learner = nnx.merge(learner_def, fn_learner_state)
            fn_rngs = nnx.merge(rngs_def, fn_rngs_state)
            action = fn_learner.act(observation, fn_rngs)

            return nnx.state(fn_learner), nnx.state(fn_rngs), action
        
        self._act = jax.jit(_act, donate_argnums=(0, 1))

        def _learn_act(fn_learner_state, fn_rngs_state, transition):
            fn_learner = nnx.merge(learner_def, fn_learner_state)
            fn_rngs = nnx.merge(rngs_def, fn_rngs_state)
            
            fn_learner.learn(transition, fn_rngs)
            action = fn_learner.act(transition.next_observation, fn_rngs)

            return nnx.state(fn_learner), nnx.state(fn_rngs), action
        
        self._learn_act = jax.jit(_learn_act, donate_argnums=(0, 1))

        def _compute_metrics(fn_learner_state):
            fn_learner = nnx.merge(learner_def, fn_learner_state)
            metrics = fn_learner.metrics.compute()
            fn_learner.metrics.reset()

            return nnx.state(fn_learner), metrics
        
        self._compute_metrics = jax.jit(_compute_metrics, donate_argnums=0)
    
    def act(self, observation) -> Action:
        self.learner_state, self.rngs_state, action = self._act(self.learner_state, self.rngs_state, observation)
        return action
    
    def learn_act(self, transition) -> Action:
        self.learner_state, self.rngs_state, action = self._learn_act(self.learner_state, self.rngs_state, transition)
        return action
    
    def compute_metrics(self):
        self.learner_state, metrics = self._compute_metrics(self.learner_state)
        return metrics
    
    def get_learner(self):
        return nnx.merge(self.learner_def, self.learner_state)
    
    def get_rngs(self):
        return nnx.merge(self.rngs_def, self.rngs_state)


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

    trainer = Trainer(learner, rngs)

    state, time_step = environment.reset()
    action = trainer.act(time_step.observation)

    for global_step in range(experiment.config.environment.max_steps):
        # action = act(learner, time_step.observation, rngs)
        state, next_time_step = environment.step(state, action)
        transition = make_transition(time_step, action, next_time_step)
        action = trainer.learn_act(transition)
        time_step = next_time_step

        if (
            global_step % experiment.config.logger.log_rate
            == experiment.config.logger.log_rate - 1
        ):
            logger.log(trainer.compute_metrics(), global_step)

            # checkpointer.save(learner, global_step)

    checkpointer.save(trainer.get_learner(), experiment.config.environment.max_steps)
    checkpointer.close()
    logger.close()

    print(f"Training finished: {experiment.unique_token}")


def make_transition(
    time_step: TimeStep, action: Action, next_time_step: TimeStep
) -> Transition:
    return Transition(
        observation=time_step.observation,
        action=action,
        reward=next_time_step.reward,
        next_observation=next_time_step.observation,
        terminated=next_time_step.step_type == StepType.LAST,
        truncated=False,
    )


def record(experiment: Experiment):
    print("Recording...")
    print(experiment.config)

    environment: EnvWrapper = EnvPoolWrapper(
        experiment.config.environment.name,
        1, #experiment.config.environment.num_envs,
        experiment.environments_seed,
    )

    rngs = nnx.Rngs(
        default=experiment.default_seed,
        params=experiment.params_seed,
        action=experiment.actions_seed,
    )

    learner = create_learner(
        experiment.config.learner,
        experiment.config.environment.num_envs,
        environment.observation_spec,
        environment.action_spec,
        rngs=rngs,
    )

    with Checkpointer(experiment.checkpoints_dir) as checkpointer:
        learner = checkpointer.restore_latest(learner)

    state, time_step = environment.reset()

    trainer = Trainer(learner, rngs)

    samples = 60 * 60
    observations = np.zeros((samples, 84, 84), dtype=np.uint8)
    observations[0] = time_step.observation.agents_view[0, 3]

    for n in range(1, samples):
        action = trainer.act(time_step.observation)
        state, time_step = environment.step(state, action)
        observations[n] = time_step.observation.agents_view[0, 3]
    
    vw.save_video(observations, "output.mp4", fps=60)
