from typing import Union
import jax
import optax

from jaxrl.config import OptimizerConfig


def create_optimizer(
    optimizer_config: OptimizerConfig, update_steps: int
) -> optax.GradientTransformation:
    match optimizer_config.type:
        case "adamw":
            return optax.chain(
                optax.adamw(
                    optax.linear_schedule(
                        optimizer_config.learning_rate, 0, update_steps
                    ),
                    b1=optimizer_config.beta1,
                    b2=optimizer_config.beta2,
                    weight_decay=optimizer_config.weight_decay,
                    eps=optimizer_config.eps,
                ),
                optax.clip_by_global_norm(optimizer_config.max_norm),
            )
        case "muon":
            return optax.chain(
                muon(
                    optax.linear_schedule(
                        optimizer_config.learning_rate, 0, update_steps
                    ),
                    weight_decay=optimizer_config.weight_decay,
                    adam_weight_decay=optimizer_config.weight_decay,
                    adam_b1=optimizer_config.beta1,
                    adam_b2=optimizer_config.beta2,
                    # adam_eps_root=optimizer_config.eps
                ),
                optax.clip_by_global_norm(optimizer_config.max_norm),
            )
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.type}")


def muon(
    learning_rate: optax.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask=None,
    mu_dtype=None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    return optax.partition(
        transforms={
            "muon": optax.chain(
                optax.contrib.scale_by_muon(
                    ns_coeffs=ns_coeffs,
                    ns_steps=ns_steps,
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                    adaptive=adaptive,
                ),
                optax.add_decayed_weights(weight_decay, weight_decay_mask),
                optax.scale_by_learning_rate(learning_rate),
            ),
            "adam": optax.adamw(
                learning_rate=learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                eps_root=adam_eps_root,
                weight_decay=adam_weight_decay,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
            ),
        },
        param_labels=lambda params: jax.tree.map_with_path(
            lambda path, x: "muon" if muon_param(path, x) else "adam", params
        ),
    )


def muon_param(path, x):
    key = path[0].key
    if (
        key == "value_head"
        or key == "action_encoder"
        or key == "reward_encoder"
        or key == "obs_encoder"
        or key == "obs_decoder"
    ):
        return False

    return x.ndim == 2
