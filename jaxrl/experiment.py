import datetime
from pydantic import BaseModel
import random
import string
import subprocess
import fsspec

from jaxrl.config import Config, load_config
from jaxrl.logger import JaxLogger


class ExperimentMeta(BaseModel):
    start_time: datetime.datetime
    git_hash: str


class Experiment:
    def __init__(self, unique_token: str, config: Config, meta: ExperimentMeta, base_dir: str) -> None:
        self.unique_token = unique_token
        self.config = config
        self.meta = meta
        self.base_dir = base_dir

        self.experiment_dir = f"{base_dir}/{self.unique_token}"
        self.config_path = f"{self.experiment_dir}/{config.json}"
        self.meta_path = f"{self.experiment_dir}/{meta.json}"

        random.seed(self.config.seed)
        self.environments_seed = random.getrandbits(31)

        self.default_seed = random.getrandbits(31)
        self.params_seed = random.getrandbits(31)
        self.actions_seed = random.getrandbits(31)

    def setup_experiment(self) -> None:
        fs, path = fsspec.core.url_to_fs(self.experiment_dir)
        fs.mkdir(self.experiment_dir, exist_ok=True)
        fs.mkdir(self.checkpoints_dir, exist_ok=True)

        config_str = self.config.model_dump_json(indent=2)
        with fsspec.open(self.config_path, "w") as f:
            f.write(config_str)

        meta_str = self.meta.model_dump_json()
        with fsspec.open(self.meta_path, "w") as f:
            f.write(meta_str)

    def create_logger(self) -> JaxLogger:
        return JaxLogger(self.config.logger, self.unique_token)

    @property
    def checkpoints_dir(self) -> str:
        return f"{self.experiment_dir}/checkpoints"

    @classmethod
    # Add base_dir here as well
    def from_config(cls, unique_token: str, config: Config, base_dir: str) -> "Experiment":
        experiment = cls(
            unique_token,
            config,
            ExperimentMeta(start_time=datetime.datetime.now(), git_hash=get_git_hash()),
            base_dir, # Pass base_dir to the constructor
        )
        experiment.setup_experiment()
        return experiment

    @classmethod
    # The entry point now also needs the base_dir
    def from_config_file(cls, config_file: str, base_dir: str) -> "Experiment":
        config = load_config(config_file)
        return cls.from_config(generate_unique_token(), config, base_dir)

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
