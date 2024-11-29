from jax import numpy as jnp
from flax import nnx
import optax

from jaxrl.config import LearnerConfig, ModelConfig, OptimizerConfig
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec, ActionSpec
from jaxrl.networks import (
    ContinuousActionHead,
    DiscreteActionHead,
    FeedForwardActorCritic as ActorCritic,
    FeedForwardValueNet as Critic,
    FeedForwardActor as Actor,
    MlpTorso,
)
from jaxrl.systems.actor_critic import ActorCriticLearner


def create_actor_head(
    action_space: ActionSpec,
    hidden_dim: int,
    *,
    dtype: jnp.dtype,
    param_dtype: jnp.dtype,
    rngs: nnx.Rngs
):
    if isinstance(action_space, DiscreteActionSpec):
        return DiscreteActionHead(
            hidden_dim,
            action_space.num_actions,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
    else:
        return ContinuousActionHead(
            hidden_dim,
            action_space.shape[0],
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )


def create_mlp_model(
    model_config: ModelConfig,
    observation_space: ObservationSpec,
    action_space: ActionSpec,
    *,
    rngs: nnx.Rngs
):
    dtype = jnp.dtype(model_config.dtype)
    param_dtype = jnp.dtype(model_config.param_dtype)

    actor_torso = MlpTorso(
        observation_space.shape[0],
        model_config.hidden_size,
        model_config.activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor_head = create_actor_head(
        action_space,
        model_config.hidden_size[-1],
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor = Actor(actor_torso, actor_head)

    critic_torso = MlpTorso(
        observation_space.shape[0],
        model_config.hidden_size,
        model_config.activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    critic = Critic(critic_torso, rngs=rngs)

    actor_critic = ActorCritic(actor, critic)
    return actor_critic


def create_optimizer(optimizer_config: OptimizerConfig) -> optax.GradientTransformation:
    return optax.adamw(
        optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        b1=optimizer_config.beta1,
        b2=optimizer_config.beta2,
    )


def create_learner(
    learner_config: LearnerConfig,
    num_envs: int,
    observation_space: ObservationSpec,
    action_space: ActionSpec,
    *,
    rngs: nnx.Rngs
):
    model = create_mlp_model(
        learner_config.model, observation_space, action_space, rngs=rngs
    )
    optimizer = create_optimizer(learner_config.optimizer)

    learner = ActorCriticLearner(
        model,
        optimizer,
        agents_shape=(num_envs,),
        discount=learner_config.discount,
        actor_coefficient=learner_config.actor_coefficient,
        critic_coefficient=learner_config.critic_coefficient,
        entropy_coefficient=learner_config.entropy_coefficient,
    )

    return learner
