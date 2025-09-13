import os
import typer
import jax
from jaxrl.experiment import Experiment
from jaxrl.play import play_from_config, play_from_run
from jaxrl.train import train_run
from jaxrl.benchmark import main as benchmark_main
from jaxrl.eval import main as eval_main
import shutil


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def enjoy(
    name: str,
    human: bool = False,
    pov: bool = False,
    seed: int = 0,
    selector: str | None = None,
    video_path: str | None = typer.Option(
        None, help="Path to save video; if omitted, no video is recorded."
    ),
):
    play_from_run(name, human, pov, seed, selector, video_path)


@app.command()
def play(
    config: str = "./config/return.json",
    human: bool = False,
    pov: bool = False,
    seed: int = 0,
    selector: str | None = None,
    video_path: str | None = typer.Option(
        None, help="Path to save video; if omitted, no video is recorded."
    ),
):
    play_from_config(config, human, pov, seed, selector, video_path)


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


app.command("benchmark")(benchmark_main)
app.command("eval")(eval_main)


if __name__ == "__main__":
    app()
