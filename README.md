## partially observable multiagent rl


This project trains transformers with PPO to solve partially obserbable multi-agent enviroments.


## Setup

uv sync --extra cuda

## Run training

uv run pmarl train --config ./config/return_baseline.json

## View training run

uv run pmarl enjoy "run_name"

## Enviroments

Variable n-back
Grid Return
Grid Return Communication
