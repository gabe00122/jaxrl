import jax
from jax import numpy as jnp
from jax.typing import DTypeLike

from flax import nnx
import tensorflow_probability.substrates.jax.distributions as tfd


from jaxrl.transformer.transformer import TransformerBlock
from jaxrl.types import Observation

class TransformerActorCritic(nnx.Module):
    def __init__(self,
        input_size: int,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_features: int,
        ffn_size: int,
        *,
        activation: str,
        use_bias: bool = True,
        glu: bool = True,
        gtrxl_gate: bool = True,
        kernel_init: nnx.Initializer,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs
    ):
        self.obs_encoder = nnx.Linear(
            input_size + 2,
            hidden_features,
            use_bias=use_bias,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(num_heads, hidden_features, ffn_size, activation=activation, glu=glu, gtrxl_gate=gtrxl_gate, kernel_init=kernel_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            )
        
        self.output_norm = nnx.LayerNorm(
            hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.value_head = nnx.Linear(
            hidden_features,
            1,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.action_head = nnx.Linear(
            hidden_features,
            action_dim,
            use_bias=use_bias,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
    
    def create_kv_cache(self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None):
        for layer in self.layers:
            layer.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(self, observation: Observation, use_kv_cache: bool) -> tuple[jax.Array, tfd.Distribution]:
        net_input = jnp.concatenate((
            observation.agents_view,
            observation.last_action,
            observation.last_reward
        ), axis=-1)

        x = self.obs_encoder(net_input)
        for layer in self.layers:
            x = layer(x, observation.time_steps, use_kv_cache)
        
        x = self.output_norm(x)

        action_logits = self.action_head(x)
        value = self.value_head(x)

        return value, tfd.Categorical(action_logits)
