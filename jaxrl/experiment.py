from calendar import c
import datetime
from math import e
from pathlib import Path
from pydantic import BaseModel
import random
import string
import subprocess

from jaxrl.config import Config, load_config


class ExperimentMeta(BaseModel):
    start_time: datetime.datetime
    git_hash: str


class Experiment:
    def __init__(self, unique_token: str, config: Config, meta: ExperimentMeta) -> None:
        self.unique_token = unique_token
        self.config = config
        self.meta = meta

        base_dir = Path("./results")
        self.experiment_dir = base_dir / self.unique_token
        self.config_path = self.experiment_dir / "config.json"
        self.meta_path = self.experiment_dir / "meta.json"

    def setup_experiment(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        config_str = self.config.model_dump_json()
        self.config_path.write_text(config_str)

        meta_str = self.meta.model_dump_json()
        self.meta_path.write_text(meta_str)

    @classmethod
    def load(cls, unique_token: str) -> "Experiment":
        base_dir = Path("./results")
        experiment_dir = base_dir / unique_token
        config_path = experiment_dir / "config.json"
        meta_path = experiment_dir / "meta.json"

        config = load_config(config_path)
        meta = ExperimentMeta.model_validate_json(meta_path.read_text())

        return cls(unique_token, config, meta)

    @classmethod
    def from_config(cls, unique_token: str, config: Config) -> "Experiment":
        experiment = cls(
            unique_token,
            config,
            ExperimentMeta(start_time=datetime.datetime.now(), git_hash=get_git_hash()),
        )
        experiment.setup_experiment()

        return experiment

    @classmethod
    def from_config_file(cls, config_file: Path) -> "Experiment":
        config = load_config(config_file)
        return cls.from_config(generate_unique_token(), config)


def generate_unique_token() -> str:
    adjectives = ["quick", "lazy", "sleepy", "noisy", "hungry"]
    nouns = ["fox", "dog", "cat", "mouse", "bear"]
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    unique_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{adjective}-{noun}-{unique_id}"


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except subprocess.CalledProcessError:
        return ""
