## JaxRL

* A actor critic implementation written with flax/nnx
* Plans to implement more RL algorithms

Inspired by [mava](https://github.com/instadeepai/Mava)

![lunar lander agent demo](/videos/rl-video-episode-5.mp4)

## Installation

Install python 3.12
Install poetry
Install a c++ compiler for gymnasium[box2d]
Install ffmpeg for (this might be installed automatically on first use)

```bash
poetry install
```

## Training

```bash
poetry run python -m jaxrl.envs.gym
```

videos will be rendered to ./videos
