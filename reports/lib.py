import json
import os
from pathlib import Path

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_experiment(experiment_dir: str) -> tuple[dict, list[dict]]:
    """Load config and JSONL logs from an experiment directory."""
    config_path = os.path.join(experiment_dir, "config.json")
    logs_path = os.path.join(experiment_dir, "logs.jsonl")

    with open(config_path) as f:
        config = json.load(f)

    logs = []
    with open(logs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    return config, logs


def find_experiments(
    filters: dict | None = None,
    group_by: str | None = None,
    results_dir: str | None = None,
) -> dict | list:
    """Find experiments matching filters, optionally grouped.

    Args:
        filters: Dict of dotted config paths to required values.
                 e.g. {"environment.env_type": "find_return", "seed": 42}
        group_by: Dotted config path to group results by.
                  e.g. "learner.model.num_layers"
        results_dir: Path to results directory. Defaults to <project>/results.

    Returns:
        If group_by is set: dict mapping group values to lists of experiment dicts.
        Otherwise: flat list of experiment dicts.
    """
    if results_dir is None:
        results_dir = str(project_root() / "results")
    if filters is None:
        filters = {}

    def get_nested(d: dict, path: str):
        for key in path.split("."):
            if not isinstance(d, dict):
                return None
            d = d.get(key)
        return d

    experiments = [] if group_by is None else {}

    for dirname in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, dirname)
        config_path = os.path.join(exp_dir, "config.json")
        logs_path = os.path.join(exp_dir, "logs.jsonl")

        if not os.path.isfile(config_path) or not os.path.isfile(logs_path):
            continue

        try:
            config, logs = load_experiment(exp_dir)
        except (json.JSONDecodeError, KeyError):
            continue

        if not all(get_nested(config, k) == v for k, v in filters.items()):
            continue

        entry = {"name": dirname, "config": config, "logs": logs}

        if group_by is None:
            experiments.append(entry)
        else:
            key = get_nested(config, group_by)
            experiments.setdefault(key, []).append(entry)

    return experiments


def extract_metrics(logs: list[dict]) -> dict[str, np.ndarray]:
    """Extract all metrics from JSONL logs into numpy arrays."""
    if not logs:
        return {}

    keys = logs[0].keys()
    result = {k: [] for k in keys}

    for entry in logs:
        for k in keys:
            result[k].append(entry.get(k, 0.0))

    return {k: np.array(v) for k, v in result.items()}


def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
