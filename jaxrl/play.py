import pygame
from jax import numpy as jnp
from flax import nnx


from jaxrl.checkpointer import Checkpointer
from jaxrl.envs.env_config import create_env
from jaxrl.envs.gridworld.renderer import GridworldClient
from jaxrl.experiment import Experiment
from jaxrl.model.network import TransformerActorCritic
from jaxrl.train import add_seq_dim


def get_action_from_keypress():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        return 0
    elif keys[pygame.K_s]:
        return 2
    elif keys[pygame.K_a]:
        return 3
    elif keys[pygame.K_d]:
        return 1

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
    run_name: str, human_control: bool, seed: int, selector: str | None = None
):
    experiment = Experiment.load(run_name, "results")
    play(experiment, human_control, seed, selector, True)


def play_from_config(
    config_name: str, human_control: bool, seed: int, selector: str | None = None
):
    experiment = Experiment.from_config_file(config_name, "", create_directories=False)
    play(experiment, human_control, seed, selector, False)


def play(
    experiment,
    human_control: bool,
    seed: int,
    selector: str | None = None,
    load: bool = True,
):
    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps, selector=selector)
    rngs = nnx.Rngs(default=seed)

    model = load_policy(experiment, env, max_steps, load, rngs)

    client = GridworldClient(env)
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
    client.render(env_state, timestep)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        human_action = get_action_from_keypress()
        if human_action is not None or not human_control:
            actions, kv_cache = sample_actions(timestep, kv_cache, rngs)
            if human_control:
                actions = actions.at[0].set(human_action)

            env_state, timestep = step(env_state, actions, rngs)
            print(timestep.last_reward[0].item())

        client.render(env_state, timestep)
