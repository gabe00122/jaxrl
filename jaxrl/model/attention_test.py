from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx

from jaxrl.model.attention import AttentionBlock


@nnx.jit
def inference_output(layer: AttentionBlock, xs: jax.Array):
    batch, seq, dim = xs.shape

    kv_cache = layer.create_kv_cache(batch)
    seq = jnp.arange(seq)

    @partial(nnx.scan, in_axes=(nnx.Carry, 1, 0), out_axes=(nnx.Carry, 1))
    def step(kv_cache, x, i):
        i = jnp.full((batch, 1), i)
        y, kv_cache = layer(x[:, None, :], i, kv_cache)

        return kv_cache, y[:, 0, ...]

    kv_cache, ys = step(kv_cache, xs, seq)
    return ys


@nnx.jit
def train_output(layer: AttentionBlock, xs: jax.Array):
    batch, seq, dim = xs.shape
    seq = jnp.arange(seq)
    ys, _ = layer(xs, seq[None, :], None)
    return ys


def test_impl():
    rngs = nnx.Rngs(random.PRNGKey(0))

    batch = 2
    seq_length = 16
    d_model = 32
    sliding_window_size = 8

    layer = AttentionBlock(
        d_model=d_model,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        max_seq_length=sliding_window_size,
        dtype=jnp.bfloat16,
        rngs=rngs,
    )

    data_key = rngs.data()
    xs = jax.random.uniform(data_key, (batch, seq_length, d_model), maxval=100)

    infer_ys = inference_output(layer, xs)
    train_ys = train_output(layer, xs)

    print(infer_ys.shape)
    print(train_ys.shape)

    # print(infer_ys)

    print(jnp.isclose(infer_ys, train_ys)[0] * 1.0)

    # print(f"test passes: {test_pass.item()}")


if __name__ == "__main__":
    test_impl()
