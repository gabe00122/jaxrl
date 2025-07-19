import jax
from jax import numpy as jnp, random
import numpy as np
from matplotlib import pyplot as plt

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, *, tileable=(False, False), interpolant=interpolant, rng_key
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = jnp.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = random.uniform(rng_key, (res[0]+1, res[1]+1), maxval=2*jnp.pi)
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return jnp.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def main():
    key = random.PRNGKey(0)
    noise = jax.jit(generate_perlin_noise_2d, static_argnums=(0, 1))((40, 40), (4, 4), rng_key=key)
    noise = noise > 0.3

    np_noise = np.asarray(noise)

    # --- Matplotlib plotting code ---
    plt.figure(figsize=(8, 8))
    plt.imshow(np_noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.title("2D Perlin Noise")
    plt.axis('off') # Hide the axes for a cleaner look
    plt.savefig("perlin_noise.png")

    # It's also good practice to close the figure to free up memory
    plt.close()


if __name__ == "__main__":
    main()
