from typing import Callable, Sequence

import jax.lax
from flax import nnx
from jax import numpy as jnp

from jaxrl.types import Observation, Action
from jaxrl.systems.types import Transition
from jaxrl.networks import FeedForwardActorCritic as ActorCritic
from jaxrl.metrics.cumulative_reward import CumulativeReward


class ActorCriticLearner(nnx.Optimizer):
    def __init__(
        self,
        model: ActorCritic,
        tx,
        agents_shape: Sequence[int],
        discount: float | Callable[[int], float],
        actor_coefficient: float | Callable[[int], float],
        critic_coefficient: float | Callable[[int], float],
        entropy_coefficient: float | Callable[[int], float],
    ):
        self.model = model
        self.discount = discount
        self.actor_coefficient = actor_coefficient
        self.critic_coefficient = critic_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            actor_loss=nnx.metrics.Average("actor_loss"),
            critic_loss=nnx.metrics.Average("critic_loss"),
            entropy_loss=nnx.metrics.Average("entropy_loss"),
            td_error=nnx.metrics.Average("td_error"),
            target=nnx.metrics.Average("target"),
            value=nnx.metrics.Average("value"),
            reward=nnx.metrics.Average("reward"),
            cumulative_reward=CumulativeReward(agents_shape),
        )
        self.global_step = nnx.Variable(jnp.uint32(0))

        super().__init__(model, tx)

    def act(self, observation: Observation, rngs: nnx.Rngs) -> Action:
        policy = self.model.actor(observation)

        action_seed = rngs.action()
        action = policy.sample(seed=action_seed)

        return action

    def learn(self, transition: Transition, rngs: nnx.Rngs):
        grad, metrics = nnx.grad(self._loss, has_aux=True)(self.model, transition, rngs)

        metrics |= {"done": transition.terminated | transition.truncated}

        self.metrics.update(**metrics)
        super().update(grad)

        self.global_step.value += 1

    def _loss(self, model: ActorCritic, transition: Transition, rngs: nnx.Rngs):
        step = self.global_step.value
        discount = self.discount(step) if callable(self.discount) else self.discount
        critic_coefficient = (
            self.critic_coefficient(step)
            if callable(self.critic_coefficient)
            else self.critic_coefficient
        )
        actor_coefficient = (
            self.actor_coefficient(step)
            if callable(self.actor_coefficient)
            else self.actor_coefficient
        )
        entropy_coefficient = (
            self.entropy_coefficient(step)
            if callable(self.entropy_coefficient)
            else self.entropy_coefficient
        )

        value, policy = model(transition.observation)
        next_value = jax.lax.stop_gradient(model.critic(transition.next_observation))

        # if done is true then the next state should be ignored
        target = transition.reward + discount * next_value * (
            1.0 - transition.terminated
        )

        temporal_difference_error = target - value

        # Critic loss
        critic_loss = temporal_difference_error**2

        # Actor loss
        action_probability = policy.log_prob(transition.action)
        actor_loss = -(
            action_probability * jax.lax.stop_gradient(temporal_difference_error)
        )

        # Entropy regularization
        entropy_seed = rngs.entropy()
        entropy_loss = -policy.entropy(seed=entropy_seed)

        loss = (
            critic_coefficient * critic_loss
            + actor_coefficient * actor_loss
            + entropy_coefficient * entropy_loss
        )

        # ignore truncated transitions
        loss *= 1.0 - transition.truncated
        loss = jnp.mean(loss)

        metrics = {
            "loss": loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
            "td_error": temporal_difference_error,
            "target": target,
            "value": value,
            "reward": transition.reward,
        }

        return loss, metrics
