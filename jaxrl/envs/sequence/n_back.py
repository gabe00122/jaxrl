import random
import jax
from jax import numpy as jnp
import pygame
from functools import cached_property

from typing import NamedTuple, Literal
from pydantic import BaseModel, ConfigDict

from jaxrl.envs.environment import Environment, TimeStep, StepType
from jaxrl.envs.specs import ObservationSpec, DiscreteActionSpec


class NBackMemoryState(NamedTuple):
    data: jax.Array
    labels: jax.Array
    position: jax.Array
    rewards: jax.Array


class NBackMemory(Environment[NBackMemoryState]):
    def __init__(self, max_n: int, max_value: int, length: int) -> None:
        self.max_n = max_n
        self.max_value = max_value
        self.length = length

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(shape=(self.max_value,), dtype=jnp.float32)

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=2)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 1

    def reset(self, rng_key: jax.Array) -> tuple[NBackMemoryState, TimeStep]:
        rng_key, n_key = jax.random.split(rng_key, 2)

        n = jax.random.randint(n_key, (), 0, self.max_n, dtype=jnp.int32)

        data = jax.random.randint(
            rng_key, (self.length,), 0, self.max_value, dtype=jnp.int32
        )
        match = jnp.equal(jnp.roll(data, n), data)
        mask = jnp.arange(self.length) >= n
        labels = jnp.where(mask, match, False)

        position = jnp.array(0, dtype=jnp.int32)

        state = NBackMemoryState(data, labels, position, jnp.float32(0.0))

        initial_timestep = self.encode_observation(
            state,
            jnp.array(
                0, dtype=jnp.int32
            ),  # Action -1 to indicate no previous action, edit 0 for now
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(StepType.FIRST.value, dtype=jnp.int32),  # Pass as JAX array
        )
        return state, initial_timestep

    def step(
        self, state: NBackMemoryState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[NBackMemoryState, TimeStep]:
        # Determine reward - only give reward if there's an n-back value to compare against
        action = action.squeeze(axis=0)

        reward = jnp.where(
            action == state.labels[state.position].astype(jnp.int32), 1.0, 0.0
        )

        # Update position
        next_position = state.position + 1

        new_state = state._replace(
            position=next_position, rewards=state.rewards + reward
        )

        done = next_position >= self.length

        # Ensure current_step_type is a JAX array of int32
        current_step_type = jnp.where(
            done,
            jnp.array(StepType.LAST.value, dtype=jnp.int32),
            jnp.array(StepType.MID.value, dtype=jnp.int32),
        )

        timestep = self.encode_observation(new_state, action, reward, current_step_type)
        return new_state, timestep

    def encode_observation(
        self,
        state: NBackMemoryState,
        last_action: jax.Array,
        last_reward: jax.Array,
        step_type: jax.Array,
    ) -> TimeStep:
        current_value = state.data[state.position]

        obs = jax.nn.one_hot(current_value, self.max_value, dtype=jnp.float32)
        action_mask = jnp.ones((2,), dtype=jnp.bool_)

        return TimeStep(
            action_mask=action_mask[None, ...],
            obs=obs[None, ...],
            time=state.position[
                None, ...
            ],  # This is the current step number / position
            last_action=last_action[None, ...],
            last_reward=last_reward[None, ...],
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: NBackMemoryState):
        # No cumulative reward tracked; keep interface consistent
        return {"rewards": jnp.float32(0.0)}


class NBackMemoryClient:
    def __init__(self, env: NBackMemory) -> None:
        self.env = env
        self.screen_width = 800
        self.screen_height = 600
        self.cell_size = 60
        self.font_size = 32
        self.screen = None
        self.font = None
        self.clock = None

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption(f"{self.env.n}-Back Memory Task")
        if self.font is None:  # Ensure font is also initialized if screen is
            pygame.font.init()  # Ensure pygame.font is initialized
            self.font = pygame.font.Font(None, self.font_size)
        if self.clock is None:  # Ensure clock is also initialized
            self.clock = pygame.time.Clock()

    def render(self, state: NBackMemoryState, ts: TimeStep) -> None:
        self._init_pygame()

        # Assertions to help linter and ensure Pygame objects are initialized
        assert self.screen is not None, "Pygame screen was not initialized."
        assert self.font is not None, "Pygame font was not initialized."

        self.screen.fill((30, 30, 30))  # Dark background

        position = ts.time[0].item()
        action_display = ts.last_action[0].item()
        reward_display = ts.last_reward[0].item()

        # Display game sequence
        history_text = self.font.render("History:", True, (200, 200, 200))
        self.screen.blit(history_text, (10, 10))

        start_x = 10
        start_y = 40

        for i, val in enumerate(state.data):
            color = (150, 150, 150)  # Default history color
            if i == position:
                color = (255, 255, 0)  # Yellow for current position

            val_text = self.font.render(str(val.item()), True, color)
            pos_x = start_x + i * (self.cell_size // 2)
            self.screen.blit(val_text, (pos_x, start_y))

        # Display current number prominently
        if position < self.env.length:
            current_val = state.data[position].item()
            current_val_text = self.font.render(
                f"Current Value: {current_val}", True, (255, 255, 255)
            )
            self.screen.blit(current_val_text, (10, start_y + 30))

            # Display N-back value if applicable
            if position >= self.env.n:
                n_back_val = state.data[position - self.env.n].item()
                n_back_text = self.font.render(
                    f"{self.env.n}-Back Value: {n_back_val}", True, (100, 100, 200)
                )
                self.screen.blit(n_back_text, (10, start_y + 60))

                correct_label = state.labels[position].item()
                label_text = self.font.render(
                    f"Correct: {'Match' if correct_label else 'No Match'}",
                    True,
                    (0, 255, 0) if correct_label else (255, 100, 100),
                )
                self.screen.blit(label_text, (10, start_y + 90))

        # Display instructions
        instruction_text = self.font.render(
            "Is it a match? (Left: No, Right: Yes)", True, (200, 200, 200)
        )
        self.screen.blit(instruction_text, (10, self.screen_height - 90))

        action_text_str = "Your Action: "
        if action_display != -1:
            action_text_str += "Match" if action_display == 1 else "No Match"
        else:
            action_text_str += "N/A"
        action_text = self.font.render(action_text_str, True, (200, 200, 200))
        self.screen.blit(action_text, (10, self.screen_height - 60))

        reward_text_str = "Last Reward: "
        reward_text_str += f"{reward_display:.1f}"
        reward_text = self.font.render(reward_text_str, True, (200, 200, 200))
        self.screen.blit(reward_text, (10, self.screen_height - 30))

    def close(self):
        if self.screen is not None:  # Check if pygame was initialized
            pygame.quit()
            self.screen = None
            self.font = None
            self.clock = None


def demo():
    n_env = NBackMemory(n=2, max_value=5, length=20)
    client = NBackMemoryClient(n_env)

    key = jax.random.PRNGKey(random.randint(0, 1000000))

    state, ts = n_env.reset(key)

    running = True
    while running:
        client.render(state, ts)
        pygame.display.flip()

        if ts.step_type[0].item() == StepType.LAST.value:
            print("Episode finished.")
            pygame.time.wait(5000)
            running = False
            continue

        agent_action = None
        input_received = False
        while not input_received and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        agent_action = jnp.array([0], dtype=jnp.int32)
                        input_received = True
                    elif event.key == pygame.K_RIGHT:
                        agent_action = jnp.array([1], dtype=jnp.int32)
                        input_received = True

            if not running:
                break

            # Ensure clock is initialized before using it
            if client.clock is not None:
                client.clock.tick(30)

        if not running or agent_action is None:
            continue

        step_key, key = jax.random.split(key)
        state, ts = n_env.step(state, agent_action, step_key)

    client.close()


if __name__ == "__main__":
    demo()


class NBackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["nback"] = "nback"

    max_n: int = 12
    max_value: int = 2
