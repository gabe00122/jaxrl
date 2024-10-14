from collections.abc import Sequence
from typing import Callable
from functools import partial

import chex
import jax
from jax import numpy as jnp
from flax import nnx

import tensorflow_probability.substrates.jax.distributions as tfd

from jaxrl.types import Observation
from jaxrl.distributions import IdentityTransformation, TanhTransformedDistribution


# Based of mava https://github.com/instadeepai/Mava/blob/develop/mava/networks.py


class MlpTorso(nnx.Module):
    def __init__(
        self,
        observation_dim: int,
        layer_sizes: Sequence[int],
        activation: str,
        *,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.observation_size = observation_dim
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.activation_fn = _parse_activation_fn(self.activation)

        self.layers = []

        in_features = observation_dim
        for out_features in self.layer_sizes:
            linear = nnx.Linear(
                in_features,
                out_features,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=nnx.initializers.he_normal(),
                # kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2)),
                rngs=rngs,
            )
            self.layers.append(linear)
            in_features = out_features

    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)

        return x


class DiscreteActionHead(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int | Sequence[int],
        *,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.action_layer = nnx.LinearGeneral(
            input_dim, action_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(
        self, obs_embedding: chex.Array, observation: Observation
    ) -> tfd.TransformedDistribution:
        actor_logits = self.action_layer(obs_embedding)

        if observation.action_mask is not None:
            actor_logits = jnp.where(
                observation.action_mask,
                actor_logits,
                jnp.finfo(self.dtype).min,
            )

        #  We transform this distribution with the `Identity()` transformation to
        # keep the API identical to the ContinuousActionHead.
        return IdentityTransformation(distribution=tfd.Categorical(logits=actor_logits))


class ContinuousActionHead(nnx.Module):
    """
    Outputs between -1 and 1
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        min_scale: float = 1e-3,
        *,
        dtype,
        param_dtype,
        input_specific_std: bool = True,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.min_scale = min_scale

        self.dtype = dtype
        self.param_dtype = param_dtype
        self.input_specific_std = input_specific_std

        linear = partial(
            nnx.Linear,
            self.input_dim,
            self.action_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nnx.initializers.orthogonal(0.01),
        )

        self.mean = linear(rngs=rngs)
        self.log_std = (
            linear(rngs=rngs)
            if input_specific_std
            else nnx.Param(jnp.zeros(self.action_dim))
        )

    def __call__(
        self, obs_embedding: chex.Array, observation: Observation
    ) -> tfd.TransformedDistribution:
        """Action selection for continuous action space environments.

        Args:
        ----
            obs_embedding (chex.Array): Observation embedding.
            observation (Observation): Observation object.

        Returns:
        -------
            tfd.Independent: Independent transformed distribution.

        """
        loc = self.mean(obs_embedding)
        scale = (
            self.log_std(obs_embedding)
            if self.input_specific_std
            else self.log_std.value
        )

        scale = jax.nn.softplus(scale) + self.min_scale

        distribution = tfd.Normal(loc=loc, scale=scale)

        return tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )


class FeedForwardActor(nnx.Module):
    """Feed Forward Actor Network."""

    def __init__(
        self,
        torso: Callable[[chex.Array], chex.Array],
        action_head: Callable[[chex.Array, Observation], tfd.Distribution],
    ):
        self.torso = torso
        self.action_head = action_head

    def __call__(self, observation: Observation) -> tfd.Distribution:
        """Forward pass."""
        obs_embedding = self.torso(observation.agents_view)
        return self.action_head(obs_embedding, observation)


class FeedForwardValueNet(nnx.Module):
    """Feedforward Value Network. Returns the value of an observation."""

    def __init__(
        self,
        torso: Callable[[chex.Array], chex.Array],
        embedding_dim: int,
        *,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.torso = torso
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.output = nnx.Linear(
            self.embedding_dim,
            1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            # kernel_init=nnx.initializers.orthogonal(1.0),
            rngs=rngs,
        )

    def __call__(self, observation: Observation) -> chex.Array:
        """Forward pass."""
        observation = observation.agents_view

        critic_output = self.torso(observation)
        critic_output = self.output(critic_output)

        return jnp.squeeze(critic_output, axis=-1)


class FeedForwardActorCritic(nnx.Module):
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

    def __call__(self, observation: Observation) -> tuple[chex.Array, tfd.Distribution]:
        value = self.critic(observation)
        policy = self.actor(observation)

        return value, policy


def _parse_activation_fn(activation_name: str) -> Callable[[chex.Array], chex.Array]:
    match activation_name:
        case "relu":
            return jax.nn.relu
        case "mish":
            return jax.nn.mish
        case _:
            raise f"Activation function {activation_name} not recognized"
