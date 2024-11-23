import json
from pathlib import Path
import typer

from jaxrl.config import Config
from jaxrl.experiment import Experiment

app = typer.Typer()


@app.command()
def train(config_file: Path):
    Experiment.from_config_file(config_file)
    

if __name__ == "__main__":
    app()
