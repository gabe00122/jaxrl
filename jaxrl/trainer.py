from jaxrl.experiment import Experiment
from jaxrl.envs.wrapper import EnvWrapper
from jaxrl.envs.envpool import EnvPoolWrapper

class Trainer:
    pass


def train(experiment: Experiment):
    print("Training...")

    environment: EnvWrapper = EnvPoolWrapper(
        experiment.config.environment.name,
        experiment.config.environment.num_envs,
        experiment.environment_seed
    )

    print(environment.action_spec)
    print(environment.observation_spec)
    
    state, time_step = environment.reset()
    print(time_step.observation.agents_view.shape)

    print(environment.players)

