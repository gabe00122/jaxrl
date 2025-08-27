import jax
from flax import nnx
from einops import rearrange

from jaxrl.config import FlattenedObsEncoderConfig, GridCnnObsEncoderConfig, LinearObsEncoderConfig, ResCnnObsEncoderConfig
from jaxrl.envs.specs import ObservationSpec
from jaxrl.resnet import ObsEncoderCNN


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

        self.params_dtype = params_dtype
        self.num_classes = obs_spec.max_value
        self.conv0 = nnx.Conv(
            in_features=self.num_classes,
            out_features=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=16,
            out_features=32,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=output_size,
            kernel_size=(3, 3),
            padding="VALID",
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = jax.nn.one_hot(x, self.num_classes, dtype=self.params_dtype)

        x = self.conv0(x)
        x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        x = rearrange(x, "... w h c -> ... (w h c)")

        return x


class FlattenedObsEncoder(nnx.Module):
    def __init__(
        self,
        config: FlattenedObsEncoderConfig,
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

        embed_features = 4

        self.params_dtype = params_dtype
        self.num_classes = obs_spec.max_value
        in_features = embed_features * obs_spec.shape[0] * obs_spec.shape[1]

        self.embedding = nnx.Linear(
            self.num_classes,
            embed_features,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )
        self.dense = nnx.Linear(
            in_features,
            output_size,
            dtype=dtype,
            param_dtype=params_dtype,
            rngs=rngs,
        )

    def __call__(self, x) -> jax.Array:
        x = jax.nn.one_hot(x, self.num_classes, dtype=self.params_dtype)
        x = self.embedding(x)
        x = rearrange(x, "... w h c -> ... (w h c)")
        x = self.dense(x)

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
    config: LinearObsEncoderConfig | GridCnnObsEncoderConfig | ResCnnObsEncoderConfig | FlattenedObsEncoderConfig,
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
        case "res_cnn":
            return ObsEncoderCNN(rngs=rngs)
        case "grid_flattened":
            return FlattenedObsEncoder(
                config,
                obs_spec,
                output_size,
                dtype=dtype,
                params_dtype=params_dtype,
                rngs=rngs,
            )
