from pathlib import Path
import typer

from jaxrl.experiment import Experiment
from jaxrl.trainer import train as train_fn

app = typer.Typer()


@app.command()
def train(config_file: Path):
    experiment = Experiment.from_config_file(config_file)
    train_fn(experiment)


if __name__ == "__main__":
    app()
