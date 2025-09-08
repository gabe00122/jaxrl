# Partially Observable Multi-Agent RL with Transformers

This project provides a JAX-based framework for training Transformer-based agents in custom, partially observable multi-agent reinforcement learning (MARL) environments. The entire training pipeline, from environment stepping to parameter updates, is JIT-compiled for maximum performance.

🚧 This project is under active development and some things might not be stable 🚧

### ✨ Key Features

* **Transformer-based Agents**: Uses a Transformer-over-time architecture to handle partial observability.
* **High Performance**: Small models train a **10 million+** steps per second and the medium sized transformer trains at **2 million** steps per second on a single 5090.
* **Custom Environments**: Includes several challenging MARL environments designed to test memory and coordination.
* **PPO Implementation**: A clean and efficient Proximal Policy Optimization (PPO) implementation.
* **Multi-task training**: One model can be trained on multiple environments simultaneously.

* **Distributed Training**: Training can be distributed to several TPUs or GPUs (temporarily regressed)

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
```


## 💻 Usage
### Train an Agent
To start a training run, use the train command and provide a configuration file.


```bash
uv run pmarl train --config ./config/return_baseline.json
```
A unique run name will be generated for you (e.g., silly-camel-34). You will need this name to view the results.


### Watch a Trained Agent
To render an environment with a trained agent, use the enjoy command with the run_name from your training session.


```bash
uv run pmarl enjoy "silly-camel-34"
```

To play as one of the agents:
```bash
uv run pmarl enjoy "silly-camel-34" --human --pov
```
* `--human` means you control one of the agents with keyboard controls and `--pov` renders the environment from the agents point of view

If the agent was trained on multiple environments you can select the right one using the `--selector` option. These correspond to the config.json
```bash
uv run pmarl enjoy "silly-camel-34" --selector return40
```

### Test out an environment
You can test out a environment without training a model using the `play` command.

```
uv run pmarl play ./config/return_baseline.json
```
---

## 🏋️‍♂️ Environments
### Variable n-back
A single-agent memory task. The agent must determine if the current observation matches the observation from `n` steps ago. `n` is randomized and not part of the observation so must be deduced from the rewards via trial and error.

* Observation: A discrete integer value.

* Actions: [`match`, `no-match`]

* Reward: `+1` for a correct action, `0` for an incorrect action.

* Example (n=2):

  * Observations: `[A, B, A, C, B, B, D, E]`

  * Correct Actions: `[_, _, match, no-match, no-match, match, no-match, no-match]`

### Grid Return
A multi-agent 2D grid world task requiring spatial memory.

* Description: A goal is placed at a random location. When an agent finds the goal, it receives a `+1` reward, and the agent is moved to a new random location. Agents must remember goal locations and navigate around obstacles to return to them efficiently.

* Observation: A small rectangular grid centered on the agent. Agents can see each other. No absolute positions are given.

* Actions: [`move-up`, `move-right`, `move-down`, `move-left`]

* Reward: `+1` for finding a goal tile, `0` for anything else.

* Note: Agents can move through each other but not through obstacle tiles.

Agents can "dig" through obstacle tiles. Moving into an obstacle removes the tile but adds a timeout before the agent can move again.

https://github.com/user-attachments/assets/a98a9277-de7e-43a6-a0c2-d6b8856eba10

### Scouts
A multi-agent coordination task with two specialized agent types. A Harvester must first unlock a resource tile, which a Scout can then gather.

Scout: Fast-moving agents that can gather resources only after they are "unlocked."
Harvester: Slow agents (can only move every 6th turn). When a Harvester reaches a resource tile, it gets a reward and unlocks the resource, allowing Scouts to gather it.

https://github.com/user-attachments/assets/01a20e6c-6a61-47ce-947b-8f0b22e27889


### Traveling Salesman

The several way points are randomly scattered and the agents and the agents get a reward for the first time they get each flag. When all flags are taken they all reset.

### King of the Hill

Two teams of agents compete to capture random control points in the center.
