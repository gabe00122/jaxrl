from einops import rearrange
import jax
import jax.numpy as jnp
from flax import nnx

# ---------- blocks ----------

class ResBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=(1, 1), padding="SAME", rngs=rngs)
        self.ln1 = nnx.LayerNorm(out_ch, rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), strides=(1, 1), padding="SAME", rngs=rngs)
        self.ln2 = nnx.LayerNorm(out_ch, rngs=rngs)

        # if channel mismatch, learnable projection
        if in_ch != out_ch:
            self.skip = nnx.Conv(in_ch, out_ch, (1, 1), strides=(1, 1), rngs=rngs)
        else:
            self.skip = None

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.conv1(x)
        y = jax.nn.silu(self.ln1(y))
        y = self.conv2(y)
        y = self.ln2(y)
        if self.skip is not None:
            x = self.skip(x)
        return jax.nn.silu(x + y)

class DownBlock(nnx.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_ch, out_ch, (k, k), strides=(2, 2), padding="SAME", rngs=rngs)
        self.ln = nnx.LayerNorm(out_ch, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.ln(x)
        return jax.nn.silu(x)

def _even_pad_hw(x: jax.Array) -> jax.Array:
    _, H, W, _ = x.shape
    ph = 1 if (H % 2) == 1 else 0
    pw = 1 if (W % 2) == 1 else 0
    if ph or pw:
        x = jnp.pad(x, ((0,0),(0,ph),(0,pw),(0,0)), mode="edge")
    return x

# ---------- encoder ----------

class ObsEncoderCNN(nnx.Module):
    def __init__(self, embed_dim: int = 256, *, rngs: nnx.Rngs):
        # Stage 1
        self.down1 = DownBlock(1, 32, k=5, rngs=rngs)
        self.res1a = ResBlock(32, 32, rngs=rngs)
        self.res1b = ResBlock(32, 32, rngs=rngs)

        # Stage 2
        self.down2 = DownBlock(32, 64, k=3, rngs=rngs)
        self.res2a = ResBlock(64, 64, rngs=rngs)
        self.res2b = ResBlock(64, 64, rngs=rngs)

        # Stage 3
        self.down3 = DownBlock(64, 128, k=3, rngs=rngs)
        self.res3a = ResBlock(128, 128, rngs=rngs)
        self.res3b = ResBlock(128, 128, rngs=rngs)

        # Head
        self.post_ln = nnx.LayerNorm(128, rngs=rngs)
        self.proj = nnx.Linear(128, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # ensure clean halving
        # x = _even_pad_hw(x)
        b, t, _, _, _ = x.shape
        x = rearrange(x, "b t x y c -> (b t) x y c")

        x = self.down1(x)
        x = self.res1a(x); x = self.res1b(x)

        x = self.down2(x)
        x = self.res2a(x); x = self.res2b(x)

        x = self.down3(x)
        x = self.res3a(x); x = self.res3b(x)

        # Global average pool
        x = x.mean(axis=(1, 2))  # (B, 128)
        x = self.post_ln(x)
        z = self.proj(x)         # (B, embed_dim)

        z = rearrange(z, "(b t) z -> b t z", b=b, t=t)

        return z

# ---------- test ----------

if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    enc = ObsEncoderCNN(rngs=rngs)
    dummy = jnp.zeros((8, 1, 65, 55, 3), dtype=jnp.float32)
    z = enc(dummy)
    print(z.shape)  # (8, 256)
