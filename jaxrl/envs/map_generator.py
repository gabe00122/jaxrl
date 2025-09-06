from jax import numpy as jnp, random
import jax
import jaxrl.envs.gridworld.constance as GW

def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


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
    grid = (
        jnp.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )
    # Gradients
    angles = random.uniform(rng_key, (res[0] + 1, res[1] + 1), maxval=2 * jnp.pi)
    gradients = jnp.dstack((jnp.cos(angles), jnp.sin(angles)))
    if tileable[0]:
        gradients[-1, :] = gradients[0, :]
    if tileable[1]:
        gradients[:, -1] = gradients[:, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # Ramps
    n00 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = jnp.sum(jnp.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = jnp.sum(jnp.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return jnp.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def fractal_noise(width: int, height: int, res: list[int], rng_key: jax.Array):
    keys = jax.random.split(rng_key, len(res) + 1)
    amplitude_key = keys[0]
    noise_keys = keys[1:]

    amplitudes = jax.random.dirichlet(amplitude_key, jnp.ones((len(res),)))
    noise = None

    for i, r in enumerate(res):
        new_noise = generate_perlin_noise_2d((width, height), (r, r), rng_key=noise_keys[i]) * amplitudes[i]
        if noise is not None:
            noise = noise + new_noise
        else:
            noise = new_noise
    
    return noise


def generate_decor_tiles(width: int, height: int, rng_key: jax.Array):
    tile_ids = jnp.array([GW.TILE_EMPTY, GW.TILE_DECOR_1, GW.TILE_DECOR_2, GW.TILE_DECOR_3, GW.TILE_DECOR_4], dtype=jnp.int8)
    tile_probs = jnp.array([0.90, 0.04, 0.04, 0.015, 0.005])
    tiles = jax.random.choice(rng_key, tile_ids, (width, height), p=tile_probs)

    return tiles


def choose_positions(tiles: jax.Array, n: int, rng_key: jax.Array):
    """
    Chooses 'n' non-repeating empty tile position
    """
    width, height = tiles.shape
    size = width * height

    available_tiles = (tiles == GW.TILE_EMPTY).flatten()

    prob_mask = available_tiles / jnp.sum(available_tiles)

    choice_indices = jax.random.choice(rng_key, size, (n,), replace=False, p=prob_mask)

    choice_x = choice_indices // height
    choice_y = choice_indices % height

    return choice_x, choice_y
