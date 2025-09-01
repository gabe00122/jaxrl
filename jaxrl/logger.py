from abc import ABC, abstractmethod
import os
from pathlib import Path

from rich.console import Console
import jax
from flax import nnx
import numpy as np
from tensorboardX import SummaryWriter
from jaxrl.config import Config
from jaxrl.util import json_normalize

# import neptune
# from neptune.utils import stringify_unsupported
import wandb

Metrics = dict[str, jax.Array | nnx.Metric]


class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, unique_token: str):
        pass

    def log_dict(self, data: Metrics, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class MultiLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self.loggers = loggers

    def log_dict(self, data: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


class JaxLogger:
    def __init__(self, settings: Config, unique_token: str, console: Console):
        loggers: list[BaseLogger] = []

        if settings.logger.use_tb:
            loggers.append(TensorboardLogger(unique_token))
        if settings.logger.use_console:
            loggers.append(ConsoleLogger(unique_token, console))
        # if cfg.logger.use_neptune:
        #     loggers.append(NeptuneLogger(cfg, unique_token))
        if settings.logger.use_wandb:
            loggers.append(WandbLogger(unique_token, settings))

        self.logger = MultiLogger(loggers)

    def log(self, metrics: Metrics, step: int) -> None:
        metrics = jax.tree.map(describe, metrics)
        self.logger.log_dict(metrics, step)

    def close(self) -> None:
        self.logger.close()


class TensorboardLogger(BaseLogger):
    def __init__(self, unique_token: str) -> None:
        log_path = Path("./logs/tensorboard") / unique_token
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_path.as_posix())

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep="/")

        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class ConsoleLogger(BaseLogger):
    def __init__(self, unique_token: str, console: Console) -> None:
        self._console = console

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep=".")

        keys = data.keys()
        values = []
        for value in data.values():
            if isinstance(value, jax.Array):
                value = value.item()
            values.append(f"{value:.3f}" if isinstance(value, float) else value)

        log_str = "\n".join([f"{key}: {value}" for key, value in zip(keys, values)])
        log_str = f"step: {step}\n{log_str}"
        self._console.print(log_str)

# class NeptuneLogger(BaseLogger):
#     def __init__(self, unique_token: str):
#         self.logger = neptune.init_run(
#             project="gabe00122/sentiment-lm",
#         )

#         self.logger["config"] = dump_settings(cfg)

#     def log_dict(self, data: Metrics, step: int) -> None:
#         data = json_normalize(data, sep="/")

#         for key, value in data.items():
#             self.logger[key].log(value, step=step)

#     def close(self) -> None:
#         self.logger.stop()


class WandbLogger(BaseLogger):
    def __init__(self, unique_token: str, settings: Config):
        wandb.init(project="jaxrl", name=unique_token, config=dump_settings(settings))

    def log_dict(self, data: Metrics, step: int) -> None:
        normalized_data = json_normalize(data)

        wandb.log(normalized_data, step=step)

    def close(self) -> None:
        wandb.finish()


def describe(x: dict) -> dict:
    if not isinstance(x, jax.Array) or x.size <= 1:
        return x

    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}


def dump_settings(settings: Config):
    return settings.model_dump()
