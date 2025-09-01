from random import randint
import optuna
from rich.console import Console
import typer

from jaxrl.config import (
    Config,
    GridCnnObsEncoderConfig,
    LearnerConfig,
    LoggerConfig,
    OptimizerConfig,
    PPOConfig,
    TransformerActorCriticConfig,
    AttentionConfig,
)
from jaxrl.envs.env_config import ReturnConfig
from jaxrl.experiment import Experiment
from jaxrl.hl_gauss import HlGaussConfig
from jaxrl.train import train_run

# This needs to be updated to match the new config structure.


def objective(trial: optuna.Trial):
    config = Config(
        seed=randint(0, 100000),
        num_envs=256,
        max_env_steps=512,
        update_steps=1800 * 2,
        updates_per_jit=10,
        environment=ReturnConfig(num_agents=16),
        hl_gauss=HlGaussConfig(
            min_value=-10.0,
            max_value=10.0,
            n_logits=51,
            sigma=trial.suggest_float("sigma", 0.05, 0.15),
        ),
        learner=LearnerConfig(
            model=TransformerActorCriticConfig(
                obs_encoder=GridCnnObsEncoderConfig(),
                hidden_features=128,
                value_hidden_dim=768,
                num_layers=6,
                activation="gelu",
                norm="rms_norm",
                dtype="bfloat16",
                param_dtype="float32",
                transformer_block=AttentionConfig(
                    attention_impl="cudnn",
                    num_heads=4,
                    num_kv_heads=1,
                    head_dim=32,
                    ffn_size=768,
                    glu=True,
                ),
            ),
            optimizer=OptimizerConfig(
                type=trial.suggest_categorical("optimizer", ["adamw", "muon"]),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.0005, 0.05, log=True
                ),
                weight_decay=trial.suggest_float("weight_decay", 0.00005, 0.0002),
                eps=1e-8,
                beta1=trial.suggest_float("beta1", 0.85, 0.95),
                beta2=trial.suggest_float("beta2", 0.9, 0.999, log=True),
                max_norm=trial.suggest_float("max_norm", 0.1, 0.5),
            ),
            trainer=PPOConfig(
                trainer_type="ppo",
                minibatch_count=16,
                vf_coef=trial.suggest_float("vf_coef", 0.001, 1.0),
                obs_coef=trial.suggest_float("obs_coef", 0.000, 0.2),
                entropy_coef=trial.suggest_float(
                    "entropy_coef", 0.00005 / 2, 0.00005 * 2
                ),
                vf_clip=trial.suggest_float(
                    "vf_clip", 0.2717602880463028 / 2, 0.2717602880463028 * 2
                ),
                discount=trial.suggest_float("discount", 0.96, 0.99),
                gae_lambda=trial.suggest_float("gae_lambda", 0.90, 0.99),
            ),
        ),
        logger=LoggerConfig(use_wandb=True),
    )

    return train_run(
        experiment=Experiment.from_config(
            config=config, unique_token=f"dimoned_trial_{trial.number}"
        ),
        trial=trial,
    )


app = typer.Typer()


@app.command()
def sweep():
    """Runs an Optuna sweep."""
    storage_name = "sqlite:///jaxrl_study_v4.db"
    study_name = "jaxrl_study"

    # import optunahub
    # module = optunahub.load_module(package="samplers/auto_sampler")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        # pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=300)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    console = Console()
    console.print("Study statistics: ")
    console.print(f"  Number of finished trials: {len(study.trials)}")
    console.print(f"  Number of pruned trials: {len(pruned_trials)}")
    console.print(f"  Number of complete trials: {len(complete_trials)}")

    console.print("Best trial:")
    trial = study.best_trial

    console.print(f"  Value: {trial.value}")

    console.print("  Params: ")
    for key, value in trial.params.items():
        console.print(f"    {key}: {value}")


if __name__ == "__main__":
    sweep()
