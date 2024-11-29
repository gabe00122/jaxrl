import array
from functools import partial
import numpy as np
from jax import numpy as jnp
from flax import nnx
import optax
import gymnasium as gym

from jaxrl.types import Observation, Action
from jaxrl.networks import (
    FeedForwardActorCritic as ActorCritic,
    FeedForwardActor as Actor,
    FeedForwardValueNet as Critic,
    MlpTorso,
    DiscreteActionHead,
    ContinuousActionHead,
)
from jaxrl.systems.actor_critic import ActorCriticLearner, Transition

from jaxrl.checkpointer import Checkpointer
from jaxrl.logger import JaxLogger, LoggerConfig


def default_model(
    rngs: nnx.Rngs, observation_space: gym.Space, action_space: gym.Space
) -> ActorCritic:
    hidden_size = (128, 128, 128)
    activation = "mish"
    dtype = jnp.float32
    param_dtype = jnp.float32

    observation_dim = observation_space.shape[0]
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        continuous = False
    else:
        action_dim = action_space.shape[0]
        continuous = True

    actor_torso = MlpTorso(
        observation_dim,
        hidden_size,
        activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )
    critic_torso = MlpTorso(
        observation_dim,
        hidden_size,
        activation,
        dtype=dtype,
        param_dtype=param_dtype,
        rngs=rngs,
    )

    action_head = ContinuousActionHead if continuous else DiscreteActionHead

    actor = Actor(
        actor_torso,
        action_head(
            hidden_size[-1], action_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        ),
    )
    critic = Critic(critic_torso, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
    model = ActorCritic(actor, critic)

    return model


def create_learner(
    total_steps: int,
    agents_shape: tuple[int],
    observation_space: gym.Space,
    action_space: gym.Space,
    rngs: nnx.Rngs,
):
    model = default_model(rngs, observation_space, action_space)

    learner = ActorCriticLearner(
        model,
        optax.adamw(
            optax.linear_schedule((2**-9), 0.0, total_steps),
            weight_decay=0.001,
            b1=0.97,
            b2=0.97,
        ),
        agents_shape,
        0.99,
        0.5,
        1.0,
        0.000005,
    )

    return learner


@nnx.jit
def act(
    _learner: ActorCriticLearner, _observation: Observation, _rngs: nnx.Rngs
) -> Action:
    return _learner.act(_observation, _rngs)


@nnx.jit
def learn(_learner: ActorCriticLearner, _transition: Transition, _rngs: nnx.Rngs):
    _learner.learn(_transition, _rngs)


def convert_observation(_observation):
    _observation = jnp.asarray(_observation)
    return Observation(_observation, None)


env_name = "LunarLander-v3"
extra_args = {
    "wrappers": (
        partial(
            gym.wrappers.RescaleObservation,
            min_obs=np.array(
                [
                    -2.5,
                    -2.5,
                    -10,
                    -10,
                    -6.2831855,
                    -10,
                    -0,
                    -0,
                ],
                dtype=np.float32,
            ),
            max_obs=np.array([2.5, 2.5, 10, 10, 6.2831855, 10, 1, 1], dtype=np.float32),
        ),
    ),
    "continuous": False,
}
num_envs = 64  # it works a lot better with a batch of training data, (multiple parallel environments)
total_steps = 500_000


def main():
    logger = JaxLogger(LoggerConfig(use_tb=True, use_console=True), "lander")

    env = gym.make_vec(env_name, num_envs=num_envs, **extra_args)

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    rngs = nnx.Rngs(1)
    learner = create_learner(
        total_steps, (num_envs,), observation_space, action_space, rngs
    )

    step = jnp.zeros(num_envs, dtype=jnp.uint32)

    observation, info = env.reset()
    observation = convert_observation(observation)

    log_rate = 100

    for global_step in range(total_steps):
        action = act(learner, observation, rngs)
        next_observation, reward, terminated, truncated, info = env.step(
            np.asarray(action)
        )

        reward = jnp.asarray(reward)

        step = jnp.where(terminated | truncated, 0, step + 1)
        next_observation = convert_observation(next_observation)

        transition = Transition(
            observation, action, reward, next_observation, terminated, truncated
        )

        learn(learner, transition, rngs)

        observation = next_observation

        if global_step % log_rate == log_rate - 1:
            logger.log(learner.metrics.compute(), global_step)
            learner.metrics.reset()

    with Checkpointer("./checkpoints/lunarlander") as checkpointer:
        checkpointer.save(learner, total_steps)


def view():
    env = gym.wrappers.RecordVideo(
        gym.make(env_name, render_mode="rgb_array"),
        "videos",
        episode_trigger=lambda step: True,
    )

    observation_space = env.observation_space
    action_space = env.action_space

    rngs = nnx.Rngs(1)
    learner = create_learner(total_steps, (64,), observation_space, action_space, rngs)

    with Checkpointer("./checkpoints/lunarlander") as checkpointer:
        learner = checkpointer.restore_latest(learner)

    step = jnp.zeros(1, dtype=jnp.uint32)

    observation, info = env.reset()

    for _ in range(5000):
        action = act(learner, convert_observation(observation), rngs)

        next_observation, reward, terminated, truncated, info = env.step(
            np.asarray(action)
        )
        done = terminated | truncated

        step += 1
        observation = next_observation

        if done:
            observation, info = env.reset()
            step = jnp.zeros(1, dtype=jnp.uint32)

    print(learner.metrics.compute()["cumulative_reward"])


if __name__ == "__main__":
    main()
    view()
