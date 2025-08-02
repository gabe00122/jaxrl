# Partially Observable Multi-Agent RL with Transformers

This project provides a JAX-based framework for training Transformer-based agents in custom, partially observable multi-agent reinforcement learning (MARL) environments. The entire training pipeline, from environment stepping to parameter updates, is JIT-compiled for maximum performance.

### ✨ Key Features

* **Transformer-based Agents**: Uses a Transformer-over-time architecture to handle partial observability.
* **High Performance**: Achieves **millions of environment steps per second** training on a single NVIDIA 5090 GPU, thanks to end-to-end JIT compilation with JAX and cudnn.
* **Custom Environments**: Includes several challenging MARL environments designed to test memory and coordination.
* **PPO Implementation**: A clean and efficient Proximal Policy Optimization (PPO) implementation.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.13+
* [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver)

### Installation

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo
uv sync --extra cuda



## 🏃‍♀️ Usage
Train an Agent
To start a training run, use the train command and provide a configuration file.

Bash

uv run pmarl train --config ./config/return_baseline.json
A unique run name will be generated for you (e.g., silly-camel-34). You will need this name to view the results.

Watch a Trained Agent
To render an environment with a trained agent, use the enjoy command with the run_name from your training session.

Bash

uv run pmarl enjoy "silly-camel-34"
