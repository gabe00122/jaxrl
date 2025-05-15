from jax import numpy as jnp
from flax import nnx
import optax

from jaxrl.config import (
    CnnConfig,
    LearnerConfig,
    MlpConfig,
    ModelConfig,
    OptimizerConfig,
)
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec, ActionSpec
from jaxrl.networks import (
    CnnTorso,
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
    rngs: nnx.Rngs,
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
    mlp_config: MlpConfig,
    observation_space: ObservationSpec,
    action_space: ActionSpec,
    *,
    rngs: nnx.Rngs,
):
    dtype = jnp.dtype(model_config.dtype)
    param_dtype = jnp.dtype(model_config.param_dtype)

    actor_torso = MlpTorso(
        observation_space.shape[0],
        mlp_config.layers,
        model_config.activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor_head = create_actor_head(
        action_space,
        mlp_config.layers[-1],
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor = Actor(actor_torso, actor_head)

    critic_torso = MlpTorso(
        observation_space.shape[0],
        mlp_config.layers,
        model_config.activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    critic = Critic(critic_torso, rngs=rngs)

    actor_critic = ActorCritic(actor, critic)
    return actor_critic


def create_cnn_model(
    model_config: ModelConfig,
    cnn_config: CnnConfig,
    observation_space: ObservationSpec,
    action_space: ActionSpec,
    *,
    rngs: nnx.Rngs,
):
    dtype = jnp.dtype(model_config.dtype)
    param_dtype = jnp.dtype(model_config.param_dtype)

    actor_torso = CnnTorso(
        observation_space.shape,
        cnn_config,
        model_config.activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor_head = create_actor_head(
        action_space,
        cnn_config.output_size,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    actor = Actor(actor_torso, actor_head)

    critic_torso = CnnTorso(
        observation_space.shape,
        cnn_config,
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
        optax.linear_schedule(optimizer_config.learning_rate, 0, 7_000_000),
        b1=optimizer_config.beta1,
        b2=optimizer_config.beta2,
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.eps,
    )


def create_learner(
    learner_config: LearnerConfig,
    num_envs: int,
    observation_space: ObservationSpec,
    action_space: ActionSpec,
    *,
    rngs: nnx.Rngs,
):
    body_config = learner_config.model.body

    if body_config.type == "mlp":
        model = create_mlp_model(
            learner_config.model,
            body_config,
            observation_space,
            action_space,
            rngs=rngs,
        )
    else:
        model = create_cnn_model(
            learner_config.model,
            body_config,
            observation_space,
            action_space,
            rngs=rngs,
        )

    optimizer = create_optimizer(learner_config.optimizer)

    learner = ActorCriticLearner(
        model,
        optimizer,
        agents_shape=(num_envs,),
        discount=learner_config.discount,
        actor_coefficient=learner_config.actor_coefficient,
        critic_coefficient=learner_config.critic_coefficient,
        entropy_coefficient=optax.linear_schedule(
            learner_config.entropy_coefficient, 0, 7_000_000
        ),
    )

    return learner
