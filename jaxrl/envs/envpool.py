import envpool
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

from jax import numpy as jnp


class State:
    def __init__(self):
        pass


class Transition:
    def __init__(self):
        pass


class Action:
    def __init__(self):
        pass


class EnvPoolWrapper:
    def __init__(self):
        self.is_jittable = False

    def reset(self) -> tuple[State, Transition]:
        return State(), Transition()

    def step(self, state: State, action: Transition) -> tuple[State, Transition]:
        return state, Transition()


def main():
    env = envpool.make_gymnasium("Pong-v5", num_envs=4)
    # obs, info = env.reset()


def compare():
    gym.register_envs(ale_py)
    env = gym.make_vec("ALE/Pong-v5", num_envs=4)
    obs, info = env.reset()
    print(obs.shape)


if __name__ == "__main__":
    main()
    # compare()
