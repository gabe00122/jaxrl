import jax
import pygame
import numpy as np
from flax import nnx
from jax import numpy as jnp

from craftax.craftax.constants import (
    OBS_DIM,
    BLOCK_PIXEL_SIZE_HUMAN,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
)
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv as CraftaxEnv
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.play_craftax import CraftaxRenderer

from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.craftax_wrapper import CraftaxEnvironment
from jaxrl.envs.create import create_env
from jaxrl.experiment import Experiment
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.train import add_seq_dim
from jaxrl.utils.video_writter import save_video


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(
                f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})"
            )


def main(name: str, base_dir: str = "results", seed: int = 111):
    experiment: Experiment = Experiment.load(name, base_dir)
    max_steps = experiment.config.max_env_steps

    env = CraftaxEnvironment()
    rngs = nnx.Rngs(default=seed)


    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )

    with Checkpointer(experiment.checkpoints_url) as checkpointer:
        model = checkpointer.restore_latest(model)


    rng = rngs.env()
    rng, _rng = jax.random.split(rng)
    env_state, ts = env.reset(_rng)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env._env, env._env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state.cstate)

    # step_fn = jax.jit(env.step)

    @nnx.jit
    def step(timestep, kv_cache, env_state, rngs):
        action_key = rngs.action()
        env_key = rngs.env()
        _, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        actions = policy.sample(seed=action_key)
        actions = jnp.squeeze(actions, axis=-1)

        env_state, timestep = env.step(env_state, actions, env_key)

        return env_state, timestep, kv_cache, rngs

    # traj_history = {"state": [env_state], "action": [], "reward": [], "done": []}

    clock = pygame.time.Clock()

    carry = model.initialize_carry(1, rngs)

    frames = []

    time = 0
    while not renderer.is_quit_requested() and time < experiment.config.max_env_steps:
        # action = renderer.get_action_from_keypress(env_state)

        # if action is not None:
            # old_achievements = env_state.achievements
        env_state, ts, carry, rngs = step(ts, carry, env_state, rngs)
        # new_achievements = env_state.achievements
        # print_new_achievements(old_achievements, new_achievements)
        # if reward > 0.8:
        #     print(f"Reward: {reward}\n")

        # traj_history["state"].append(env_state)
        # traj_history["action"].append(action)
        # traj_history["reward"].append(reward)
        # traj_history["done"].append(done)

        renderer.render(env_state.cstate)
        renderer.update()
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        frames.append(img_data)

        clock.tick(5)
        time += 1
    
    frames = np.array(frames)
    frames = np.rot90(frames, -1, (1, 2))
    frames = np.flip(frames, 2)
    save_video(frames, "videos/craftax.mp4", 5)

if __name__ == "__main__":
    main("noble-mouse-nq67s4")