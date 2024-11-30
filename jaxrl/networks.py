from collections.abc import Sequence, Callable
from functools import partial

import chex
import jax
from jax import numpy as jnp
from flax import nnx

import tensorflow_probability.substrates.jax.distributions as tfd

from jaxrl.config import CnnConfig
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
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.observation_size = observation_dim
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.activation_fn = parse_activation_fn(self.activation)

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

    @property
    def output_size(self) -> int:
        return self.layer_sizes[-1]


def cnn_output_size(
    dims: Sequence[int], kernel_size: Sequence[int], stride: Sequence[int]
) -> Sequence[int]:
    return [(d - k) // s + 1 for d, k, s in zip(dims, kernel_size, stride)]


class CnnTorso(nnx.Module):
    def __init__(
        self,
        observation_dims: Sequence[int],
        cnn_config: CnnConfig,
        activation: str,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.activation_fn = parse_activation_fn(activation)
        self.output_size = cnn_config.output_size

        features = 1
        dimensions = observation_dims

        self.conv_layers = []
        for layer in cnn_config.layers:
            conv = nnx.Conv(
                in_features=features,
                out_features=layer.features,
                kernel_size=layer.kernel_size,
                strides=layer.stride,
                padding="VALID",
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.conv_layers.append(conv)
            features = layer.features
            dimensions = cnn_output_size(
                dimensions, layer.kernel_size, layer.stride
            )

        self.dense = nnx.LinearGeneral(
            in_features=(*dimensions, features),
            axis=tuple(range(-len(dimensions) - 1, 0)),
            out_features=cnn_config.output_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, observation: chex.Array) -> chex.Array:
        x = observation / 255.0
        x = jnp.expand_dims(x, axis=-1)

        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation_fn(x)

        x = self.dense(x)
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
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
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
        *,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.torso = torso
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.output = nnx.Linear(
            self.torso.output_size,
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


def parse_activation_fn(activation_name: str) -> Callable[[chex.Array], chex.Array]:
    match activation_name:
        case "relu":
            return jax.nn.relu
        case "mish":
            return jax.nn.mish
        case "gelu":
            return jax.nn.gelu
        case "silu":
            return jax.nn.silu
        case "tanh":
            return jax.nn.tanh
        case _:
            raise f"Activation function {activation_name} not recognized"
