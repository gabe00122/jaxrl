import optuna

from jaxrl.config import Config, GridCnnObsEncoderConfig, LearnerConfig, ReturnConfig, TransformerActorCriticConfig


def objective(trial: optuna.Trial):
    config = Config(
        seed=0,
        num_envs=32,
        max_env_steps=256,
        update_steps=20000,
        updates_per_jit=100,
        environment=ReturnConfig(num_agents=16),
        learner=LearnerConfig(
            model=TransformerActorCriticConfig(
                obs_encoder=GridCnnObsEncoderConfig(),
                hidden_features=128,
                num_layers=3,
                activation="gelu",
                norm="layer_norm",
                dtype="bfloat16",
                param_dtype="float32",
                transformer_block=TransformerBlockConfig(
                    num_heads=4,
                    ffn_size=512,
                    glu=False,
                    gtrxl_gate=False,
                ),
            ),
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=trial.suggest_float(
                    "learning_rate", 1e-5, 1e-3, log=True
                ),
                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
                eps=1e-8,
                beta1=0.9,
                beta2=0.999,
                max_norm=trial.suggest_float("max_norm", 0.1, 1.0),
            ),
            trainer=PPOConfig(
                trainer_type="ppo",
                minibatch_count=1,
                vf_coef=trial.suggest_float("vf_coef", 0.5, 2.0),
                entropy_coef=trial.suggest_float("entropy_coef", 0.0, 0.01),
                vf_clip=trial.suggest_float("vf_clip", 0.1, 0.3),
                discount=trial.suggest_float("discount", 0.9, 0.99),
                gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            ),
        ),
        logger=LoggerConfig(use_wandb=True),
    )

    return train_run(
        experiment=Experiment.from_config(
            config=config, unique_token=f"trial_{trial.number}"
        ),
        trial=trial,
    )


@app.command()
def sweep():
    """Runs an Optuna sweep."""
    storage_name = "sqlite:///jaxrl_study.db"
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
