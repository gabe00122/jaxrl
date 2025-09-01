import os
import typer
import jax
from jaxrl.experiment import Experiment
from jaxrl.play import play_from_config, play_from_run
from jaxrl.train import train_run
import shutil


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def enjoy(name: str, human: bool = False, seed: int = 0, selector: str | None = None):
    play_from_run(name, human, seed, selector)


@app.command()
def play(name: str, human: bool = False, seed: int = 0, selector: str | None = None):
    play_from_config(name, human, seed, selector)


@app.command("train")
def train_cmd(
    config: str = "./config/return.json",
    distributed: bool = False,
    base_dir: str = "./results",
):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(config, base_dir)

    train_run(experiment)


@app.command("profile")
def profile(
    config: str = "./config/return.json",
    distributed: bool = False,
    base_dir: str = "./results",
):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(config, base_dir, create_directories=False)

    train_run(experiment, profile=True)


@app.command("clean")
def clean():
    subfolders = [f.path for f in os.scandir("./results") if f.is_dir()]

    for folder in subfolders:
        checkpoint_count = len(
            [f.path for f in os.scandir(folder + "/checkpoints") if f.is_dir()]
        )

        if checkpoint_count == 0:
            shutil.rmtree(folder)


if __name__ == "__main__":
    app()
