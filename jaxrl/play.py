from pdb import run
from xmlrpc.client import boolean
import pygame
from jax import numpy as jnp
from flax import nnx

import jaxrl.envs.gridworld.constance as GW
from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.env_config import create_env
from jaxrl.envs.gridworld.renderer import GridworldClient
from jaxrl.experiment import Experiment
from jaxrl.model.network import TransformerActorCritic
from jaxrl.train import add_seq_dim


def get_action_from_keydown(event: pygame.event.Event | None):
    """Map a just-pressed key (KEYDOWN event) to an action.

    Returns None for events that are not relevant or not KEYDOWN.
    """
    if event is None or event.type != pygame.KEYDOWN:
        return None

    key = event.key
    if key in (pygame.K_w, pygame.K_UP):
        return GW.MOVE_UP
    elif key in (pygame.K_s, pygame.K_DOWN):
        return GW.MOVE_DOWN
    elif key in (pygame.K_a, pygame.K_LEFT):
        return GW.MOVE_LEFT
    elif key in (pygame.K_d, pygame.K_RIGHT):
        return GW.MOVE_RIGHT
    elif key == pygame.K_PERIOD:
        return GW.STAY
    elif key == pygame.K_SPACE:
        return GW.PRIMARY_ACTION
    elif key == pygame.K_e:
        return GW.DIG_ACTION

    return None


def load_policy(experiment: Experiment, env, max_steps, load: bool, rngs: nnx.Rngs):
    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )

    if load:
        with Checkpointer(experiment.checkpoints_url) as checkpointer:
            model = checkpointer.restore_latest(model)

    return model


def play_from_run(
    run_name: str,
    human_control: bool,
    pov: bool,
    seed: int,
    selector: str | None = None,
    video_path: str | None = None,
):
    experiment = Experiment.load(run_name, "results")
    play(experiment, human_control, pov, seed, selector, True, video_path)


def play_from_config(
    config_name: str,
    human_control: bool,
    pov: bool,
    seed: int,
    selector: str | None = None,
    video_path: str | None = None,
):
    experiment = Experiment.from_config_file(config_name, "", create_directories=False)
    play(experiment, human_control, pov, seed, selector, False, video_path)


def play(
    experiment,
    human_control: bool,
    pov: bool,
    seed: int,
    selector: str | None = None,
    load: bool = True,
    video_path: str | None = None,
):
    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps, selector=selector)
    rngs = nnx.Rngs(default=seed)

    model = load_policy(experiment, env, max_steps, load, rngs)

    client = GridworldClient(env, fps=6)
    if human_control:
        client.renderer.focus_agent(0)

    @nnx.jit
    def sample_actions(timestep, kv_cache, rngs):
        action_key = rngs.action()
        _, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        actions = policy.sample(seed=action_key)
        actions = jnp.squeeze(actions, axis=-1)

        return actions, kv_cache

    @nnx.jit
    def step(env_state, actions, rngs):
        env_key = rngs.env()
        env_state, timestep = env.step(env_state, actions, env_key)
        return env_state, timestep

    running = True

    kv_cache = model.initialize_carry(env.num_agents, rngs=rngs)
    env_state, timestep = env.reset(rngs.env())
    if pov:
        client.render_pov(env_state, timestep, 0)
    else:
        client.render(env_state, timestep)

    cumulative_reward = 0.0

    time = 0
    while time < 512 and running:
        did_step = False
        for event in pygame.event.get():
            if client.handle_event(event):
                continue
            if event.type == pygame.QUIT:
                running = False
            if not running:
                break

            if human_control and event.type == pygame.KEYDOWN:
                human_action = get_action_from_keydown(event)
                if human_action is not None:
                    # Step immediately on just-pressed key
                    actions, kv_cache = sample_actions(timestep, kv_cache, rngs)
                    actions = actions.at[0].set(human_action)
                    env_state, timestep = step(env_state, actions, rngs)
                    if video_path is not None:
                        client.record_frame()
                    time += 1
                    did_step = True
                    reward = timestep.last_reward[0].item()
                    cumulative_reward += reward
                    print(f"reward: {reward}")
                    # Process at most one action per loop to keep input discrete
                    break

        if not running:
            break

        # If not under human control, advance continuously
        if not human_control and not did_step:
            actions, kv_cache = sample_actions(timestep, kv_cache, rngs)
            env_state, timestep = step(env_state, actions, rngs)
            if video_path is not None:
                client.record_frame()
            time += 1

            reward = timestep.last_reward[0].item()
            cumulative_reward += reward
            print(f"reward: {reward}")

        if pov:
            client.render_pov(env_state, timestep, 0)
        else:
            client.render(env_state, timestep)
    
    print(f"Cumulative reward: {cumulative_reward}")

    if video_path is not None:
        client.save_video(video_path)
