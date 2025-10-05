# Partially Observable Multi-Agent RL with Transformers

This project provides a JAX-based framework for training Transformer-based agents in custom, partially observable multi-agent reinforcement learning (MARL) environments. The custom grid environments are designed to test memory, planning, multi-task learning and cooperation.

### ✨ Key features

* **Transformer-based Agents**: Uses a Transformer-over-time architecture to handle partial observability. Uses updated transformer architecture such as rope and pre-layer norm. 
* **High Performance**: Single layer transformers train a **10 million+** steps per second and the medium sized transformer trains at **2 million** steps per second on a single 5090. Cudnn, grouped query attention and bfloat16 training significantly boost performance.
* **Custom Environments**: Includes several challenging MARL environments designed to test memory and coordination.
* **PPO Implementation**: A clean and efficient Proximal Policy Optimization (PPO) implementation.
* **Multi-task training**: One model can be trained on multiple environments simultaneously with unique task id embeddings.  
* **Snapshot League self play**: Policy snapshots can be rotated into the opponent poll during training.  
* **Trueskill Evaluations**: Policies for zero sum games can be compared using a trueskill based evaluation system and a model checkpoint pool.  
* **Discretized Value Functions**: Option to either minimize MSE or Hl-Gauss discretized values.  

---

## 🚀 Getting started

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
### Train an agent
To start a training run, use the train command and provide a configuration file.


```bash
uv run pmarl train --config ./config/return_baseline.json
```
A unique run name will be generated for you (e.g., silly-camel-34). You will need this name to view the results.


### Watch a trained agent
To render an environment with a trained agent, use the enjoy command with the run_name from your training session.

```
uv run pmarl enjoy young-shark-cff1bi --seed 5 --video-path ./videos/out.mp4
```

### Test out an environment
You can test out a environment without training a model using the `play` command.

```
uv run pmarl play ./config/return_baseline.json --pov -- human
```

The `pov` option renders from one agents point of pov
The `human` option gives you keyboard controls for one of the agents

Typical control scheme
Keyboard:
w: up
d: right
s: down
a: left
e: dig a wall
space: attack
n: cycle to the next agent

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

* Description: One or more goal tiles are placed at random locations. When an agent finds a goal, it receives a `+1` reward, and the agent is moved to a new random location. Agents must remember goal locations and navigate around obstacles to return to them efficiently. The number of goals can be configured via `num_flags` and defaults to one.

* Observation: A small rectangular grid centered on the agent. Agents can see each other. No absolute positions are given.

* Actions: [`move-up`, `move-right`, `move-down`, `move-left`]

* Reward: `+1` for finding a goal tile, `0` for anything else.

* Note: Agents can move through each other but not through obstacle tiles.

Agents can "dig" through obstacle tiles. Moving into an obstacle removes the tile but adds a timeout before the agent can move again.

https://github.com/user-attachments/assets/98cc3318-67ca-44c0-ac6e-e4537bd30ed1

### Scouts
A multi-agent coordination task with two specialized agent types. A Harvester must first unlock a resource tile, which a Scout can then gather.

Scout: Fast-moving agents that can gather resources only after they are "unlocked."
Harvester: Slow agents (can only move every 6th turn). When a Harvester reaches a resource tile, it gets a reward and unlocks the resource, allowing Scouts to gather it.

https://github.com/user-attachments/assets/01a20e6c-6a61-47ce-947b-8f0b22e27889

### Traveling salesman

The several way points are randomly scattered and the agents and the agents get a reward for the first time they get each flag. When all flags are taken they all reset.

https://github.com/user-attachments/assets/af009d24-c65e-4195-99af-0a4e703652cd

### King of the hill

Two teams of agents compete to capture random control points in the center.

https://github.com/user-attachments/assets/a3acfbbc-f26e-4c37-babb-6b146ae478c4

A multi-agent gridworld where two teams of Knights and Archers battle to capture and hold flags.

Randomly generated maps with destructible walls and central control points

Knights: melee fighters with higher HP

Archers: ranged fighters with arrows and cooldowns

Teams score points every turn for each flag they control

Agents can move, attack, dig through walls, or fire arrows

Rewards are fully team-shared, encouraging coordination

## Bonus ##

Currently training only supports episodes the entirly fit in context, this makes variable length episodes tricky to train on but you can still train on games like craftax if episodes are trunctated to fit within context.
In this case I truncated episodes to fit within 1024 steps of context and still acheived a score of 17.7% with 1b samples. In the future the kv cache at the start of the rollout could be saved and reused in training with sliding window attention to enable learning with variable length or long episodes.

https://github.com/user-attachments/assets/d667e777-c480-4b40-b190-46946d3548d5

---

## 🙏 Acknowledgements

This project was made possible thanks to:

* **Google Research Cloud TPU Program** — for providing access to TPUs that enabled training and experimentation.  
* **[Urizen Onebit Tileset](https://vurmux.itch.io/urizen-onebit-tileset)** by Vurmux — for the excellent pixel art tileset used in the gridworld environments.  
