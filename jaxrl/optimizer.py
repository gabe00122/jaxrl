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
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.type}")
