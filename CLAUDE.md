# CLAUDE.md — jaxrl Research Assistant Guide

## Project Overview

JAX/Flax multi-agent reinforcement learning framework. Trains transformer-based agents with PPO on grid-world environments from the `mapox` package.

## Commands

- **Train**: `uv run pmarl train --config ./config/<name>.json`
- **Run scripts**: `uv run python <script>.py`
- **Dependencies**: managed via `uv` (see `pyproject.toml`)

## Project Structure

```
config/          # Experiment configs (JSON)
jaxrl/           # Main package (training, models, logging, config schema)
results/         # Training outputs: results/<adjective-noun-hash>/{config.json, logs.jsonl, ...}
reports/         # Research reports and analysis
  experiments.md # Master log of all experiments (thesis, outcome, run names)
  <topic>/       # One directory per experiment, contains report.md + plots
models/          # Saved model checkpoints
```

## Config System

Configs are JSON files validated by Pydantic models in `jaxrl/config.py`. Key fields:

```json
{
  "seed": 42,              // int or "random"
  "max_env_steps": 512,    // episode length
  "update_steps": 5000,    // total training updates
  "num_envs": 512,         // parallel environments
  "learner": {
    "optimizer": { "type": "muon", "learning_rate": 0.001, ... },
    "model": {
      "hidden_features": 128,
      "num_layers": 16,     // transformer depth
      "layer": {
        "history": { "type": "attention", "num_heads": 4, "head_dim": 32, ... },
        "feed_forward": { "glu": true, "size": 768 }
      },
      "value": { "type": "hl_gauss", ... }
    },
    "trainer": { "trainer_type": "ppo", ... }
  },
  "environment": { "env_type": "find_return", ... },
  "logger": { "use_console": true, "use_jsonl": true, "use_wandb": false }
}
```

For research experiments, always set:
- `"seed": 42` (reproducibility)
- `"use_jsonl": true` (required for analysis)
- `"use_wandb": false` (avoid noise)
- `"use_console": false` (avoid noise)

## Available Environments

| env_type | Description |
|----------|-------------|
| `find_return` | Agents navigate grid to find flags/treasure and return |
| `king_hill` | King of the Hill competitive |
| `scouts` | Scouts + harvesters cooperative |
| `stealth` | Sneakers + chasers adversarial |
| `soccer` | Soccer competition |
| `craftax` | CraftAX survival (third-party wrapper) |
| `traveling_salesman` | TSP optimization |
| `multi` | Multi-task (combines multiple envs) |

Environments come from the `mapox` package (installed in .venv). Source at:
`.venv/lib/python3.13/site-packages/mapox/envs/`

Each environment defines `create_logs()` which determines what metrics appear in JSONL logs.
For find_return: `{"rewards": state.rewards}` (single scalar, logged as `env/rewards`).

## JSONL Log Format

Each line in `results/<name>/logs.jsonl`:
```json
{"step": 0, "algo/actor_loss": -0.01, "algo/value_loss": 0.5, "algo/entropy_loss": -0.06, "algo/total_loss": 0.2, "env/rewards": 1.5}
```

## Training Speed Reference (find_return, 512 envs, 8 agents, 40x40 grid)

| Layers | Steps/min | Time for 5000 steps | Steps for ~30 min |
|--------|-----------|---------------------|--------------------|
| 1      | ~111      | ~45 min             | ~3300              |
| 2      | ~89       | ~56 min             | ~2700              |
| 4      | ~62       | ~81 min             | ~1850              |
| 8      | ~38       | ~130 min            | ~1150              |
| 16     | ~22       | ~229 min            | ~660               |

These are rough estimates on the current hardware. Scale `update_steps` accordingly to hit a time budget.

## Research Workflow

### Running experiments

1. **Create configs** by copying a base config and varying the parameter under test.
   - Name configs descriptively: `<env>_<variable>_<value>.json` (e.g., `return_4_layers.json`)
   - Keep seed=42 and logger settings consistent.
2. **Create a shell script** (`<name>-train.sh`) that runs all experiment configs sequentially.
3. **Run in background** and monitor via `wc -l results/<name>/logs.jsonl`.

### 30-minute budget

Target ~30 minutes per experiment run. Use the speed reference table above to choose `update_steps`.
When comparing models of different sizes, you may need different step counts — document this clearly.

### Analysis and reporting

1. **Write analysis scripts** as `reports/<topic>/analyze.py`.
   - Load from `results/**/logs.jsonl`, filter by seed and env_type.
   - Output plots and `report.md` to the same `reports/<topic>/` directory.
   - Run with `uv run python reports/<topic>/analyze.py`.
2. **Update `reports/experiments.md`** with a row containing:
   - Date, report directory, thesis, outcome summary, and run names from results/.
3. **Track run names** (e.g., `lively-wizard-xmz694`) so experiments can be reproduced or revisited.

### Analysis script conventions

```python
# Loading experiments
def find_experiments(results_dir="results", env_type="find_return", seed=42, **filters):
    """Find experiments matching criteria, return dict grouped by the variable under test."""

# Always smooth reward curves (window=50) before plotting
# Always report both final reward AND steps-to-threshold metrics
# Save all outputs to reports/<topic>/
```
