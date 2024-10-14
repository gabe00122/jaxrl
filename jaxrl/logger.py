from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import jax
import numpy as np
from tensorboardX import SummaryWriter

from jaxrl.types import Metrics
from jaxrl.util import json_normalize


@dataclass
class LoggerConfig:
    use_console: bool = True
    use_tb: bool = False
    use_json: bool = False


# Based of mava logger https://github.com/instadeepai/Mava/blob/develop/mava/utils/logger.py


class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, cfg: LoggerConfig, unique_token: str):
        pass

    @abstractmethod
    def log_stat(self, key: str, value: float, step: int) -> None:
        pass

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep="/")

        for key, value in data.items():
            self.log_stat(key, value, step)

    def close(self) -> None:
        pass


class MultiLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: float, step: int) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step)

    def log_dict(self, data: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


class JaxLogger:
    """
    Wrapper for jax dictionaries
    """

    def __init__(self, cfg: LoggerConfig, unique_token: str):
        loggers: list[BaseLogger] = []

        if cfg.use_tb:
            loggers.append(TensorboardLogger(cfg, unique_token))
        if cfg.use_console:
            loggers.append(ConsoleLogger(cfg, unique_token))
        if cfg.use_json:
            loggers.append(JsonLogger(cfg, unique_token))

        self.logger = MultiLogger(loggers)

    def log(self, metrics: Metrics, step: int) -> None:
        metrics = jax.tree.map(describe, metrics)
        # metrics = jax.tree.map(lambda x: x.item(), metrics)
        self.logger.log_dict(metrics, step)


class TensorboardLogger(BaseLogger):
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        log_path = Path("./tensorboard") / unique_token
        self.writer = SummaryWriter(log_path.as_posix())

    def log_stat(self, key: str, value: float, step: int) -> None:
        self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class ConsoleLogger(BaseLogger):
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        pass

    def log_stat(self, key: str, value: float, step: int) -> None:
        print(f"{key}: {value:.3f}")

    def log_dict(self, data: Metrics, step: int) -> None:
        data = json_normalize(data, sep=" ")

        keys = data.keys()
        values = []
        for value in data.values():
            if isinstance(value, jax.Array):
                value = value.item()
            values.append(f"{value:.3f}" if isinstance(value, float) else value)

        log_str = " | ".join([f"{key}: {value}" for key, value in zip(keys, values)])
        print(log_str)


class JsonLogger(BaseLogger):
    def __init__(self, cfg: LoggerConfig, unique_token: str) -> None:
        pass

    def log_stat(self, key: str, value: float, step: int) -> None:
        pass

    def log_dict(self, data: Metrics, step: int) -> None:
        pass


def describe(x: dict) -> dict:
    if not isinstance(x, jax.Array) or x.size <= 1:
        return x

    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}
