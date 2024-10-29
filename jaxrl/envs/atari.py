import gymnasium
import ale_py
from flax import nnx
from jax import numpy as jnp
import matplotlib.pyplot as plt

from jaxrl.networks import CnnTorso

gymnasium.register_envs(ale_py)

env = gymnasium.make("ALE/DemonAttack-v5")

obs, info = env.reset()

torso = CnnTorso(rngs=nnx.Rngs(default=0))
x = torso(jnp.array(obs, dtype=jnp.float32))

# print(obs.shape)

imgplot = plt.imshow(X=x)
plt.show()
