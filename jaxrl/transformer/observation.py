import jax
from flax import nnx
from einops import rearrange

from jaxrl.config import GridCnnObsEncoderConfig, LinearObsEncoderConfig
from jaxrl.envs.specs import ObservationSpec


class LinearObsEncoder(nnx.Module):
    def __init__(
        self,
        config: LinearObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(
            obs_spec.shape[0],
            output_size,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        return self.linear(x)


class GridCnnObsEncoder(nnx.Module):
    def __init__(
        self,
        config: GridCnnObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert (
            obs_spec.max_value is not None
        ), "max_value must be specified in the observation spec"

        self.dtype = dtype
        self.num_classes = obs_spec.max_value
        self.conv1 = nnx.Conv(
            in_features=self.num_classes,
            out_features=16,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=16,
            out_features=output_size,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = jax.nn.one_hot(x, self.num_classes, dtype=self.dtype)

        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        x = rearrange(x, "... w h c -> ... (w h c)")

        return x


class GridCnnObsDecoder(nnx.Module):
    def __init__(
        self,
        config: GridCnnObsEncoderConfig,
        obs_spec: ObservationSpec,
        output_size: int,
        *,
        dtype,
        params_dtype,
        rngs: nnx.Rngs,
    ) -> None:
        assert (
            obs_spec.max_value is not None
        ), "max_value must be specified in the observation spec"

        self.dtype = dtype
        self.num_classes = obs_spec.max_value
        self.conv2 = nnx.ConvTranspose(
            in_features=output_size,
            out_features=16,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.conv1 = nnx.ConvTranspose(
            in_features=16,
            out_features=self.num_classes,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = rearrange(x, "... c -> ... 1 1 c")

        x = self.conv2(x)
        x = jax.nn.gelu(x)
        x = self.conv1(x)

        return x



def create_obs_encoder(
    config: LinearObsEncoderConfig | GridCnnObsEncoderConfig,
    obs_spec: ObservationSpec,
    output_size: int,
    *,
    dtype,
    params_dtype,
    rngs: nnx.Rngs,
):
    match config.obs_type:
        case "linear":
            return LinearObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
        case "grid_cnn":
            return GridCnnObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
        