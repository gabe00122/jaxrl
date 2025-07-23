import datetime as dt
import random
import string
import subprocess
import fsspec
from pydantic import BaseModel

from jaxrl.config import Config, load_config
from jaxrl.logger import JaxLogger


class ExperimentMeta(BaseModel):
    start_time: dt.datetime
    git_hash: str


class Experiment:
    def __init__(
        self,
        unique_token: str,
        config: Config,
        meta: ExperimentMeta,
        base_dir: str = "results",
    ) -> None:

        self.unique_token = unique_token
        self.config = config
        self.meta = meta

        # build the URL once
        self.experiment_url = f"{base_dir.rstrip('/')}/{self.unique_token}"
        self.checkpoints_url = f"{base_dir.rstrip('/')}/{self.unique_token}/checkpoints"

        # filesystem handle reused everywhere
        self.fs, self.root = fsspec.core.url_to_fs(self.experiment_url)

        # derived paths that *fs* understands
        self.config_path = f"{self.root}/config.json"
        self.meta_path = f"{self.root}/meta.json"
        self.ckpt_dir = f"{self.root}/checkpoints"

        # seeds
        random.seed(self.config.seed)
        self.environments_seed = random.getrandbits(31)
        self.default_seed = random.getrandbits(31)
        self.params_seed = random.getrandbits(31)
        self.actions_seed = random.getrandbits(31)

    def create_directories(self) -> None:
        """Create the directory tree and write config & metadata."""
        self.fs.makedirs(self.ckpt_dir, exist_ok=True)

        with self.fs.open(self.config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=2))

        with self.fs.open(self.meta_path, "w") as f:
            f.write(self.meta.model_dump_json(indent=2))

    def create_logger(self) -> JaxLogger:
        return JaxLogger(self.config.logger, self.unique_token)

    @classmethod
    def load(cls, unique_token: str, base_dir: str = "results") -> "Experiment":
        experiment_url = f"{base_dir.rstrip('/')}/{unique_token}"
        fs, root = fsspec.core.url_to_fs(experiment_url)

        with fs.open(f"{root}/config.json", "r") as f:
            config = load_config(f.read())

        with fs.open(f"{root}/meta.json", "r") as f:
            meta = ExperimentMeta.model_validate_json(f.read())

        return cls(unique_token, config, meta, base_dir)

    @classmethod
    def from_config(
        cls, unique_token: str, config: Config, base_dir: str = "results", create_directories: bool = True
    ) -> "Experiment":

        meta = ExperimentMeta(
            start_time=dt.datetime.now(tz=dt.timezone.utc),
            git_hash=get_git_hash(),
        )
        exp = cls(unique_token, config, meta, base_dir)

        if create_directories:
            exp.create_directories()
        return exp

    @classmethod
    def from_config_file(
        cls, config_file: str, base_dir: str = "results", create_directories: bool = True
    ) -> "Experiment":

        with fsspec.open(config_file, "r") as f:
            config = load_config(f.read())
        return cls.from_config(generate_unique_token(), config, base_dir, create_directories=create_directories)


def generate_unique_token() -> str:
    adjectives = ["quick", "lazy", "sleepy", "noisy", "hungry"]
    nouns = ["fox", "dog", "cat", "mouse", "bear"]
    return (
        f"{random.choice(adjectives)}-{random.choice(nouns)}-"
        f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"
    )


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except subprocess.CalledProcessError:
        return ""
