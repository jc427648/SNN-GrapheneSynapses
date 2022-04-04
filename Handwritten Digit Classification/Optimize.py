# https://optuna.org/
from Network import Network
from Main import train, test
import os
import joblib
import optuna
from optuna.trial import TrialState
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils


def optimize_parameters(n_output_neurons, n_trials=50):
    sampler = optuna.samplers.TPESampler(seed=0)  # To ensure reproducibility
    run = neptune.init(api_token=os.getenv("NEPTUNE_API_TOKEN"),
                       project='JCU-NICE/Updated-SNN-Optimization',
                       mode="offline")
    neptune_callback = optuna_utils.NeptuneCallback(run)
    storage_name = "sqlite:///{}.db".format(n_output_neurons)
    study = optuna.create_study(study_name=str(n_output_neurons), direction="maximize", sampler=sampler, storage=storage_name, load_if_exists=True)
    study.enqueue_trial({'tau': 0.0456969078761216,
                         'gamma': 0.00155657657413122,
                         'v_th_max': 0.0444124263767494,
                         'fixed_inhibition_current': 7.50e-5,
                         'R': 499.12
                         })
    study.optimize(lambda trial: objective(
        trial, n_output_neurons), n_trials=n_trials, callbacks=[neptune_callback])


def objective(trial, n_output_neurons):
    """ Function with unknown internals we wish to maximize.
    """
    tau = trial.suggest_float("tau", 1e-10, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 5e-10, 5e-1, log=True)
    dt = 2e-4
    image_duration = 0.05
    n_samples_train = 60000
    n_samples_validate = 10000
    log_interval = 5000
    R = trial.suggest_float("R", 400.0, 1000.0)
    v_th_max = trial.suggest_float("v_th_max", 0.02, 0.20)
    v_th_min = 10e-3
    target_activity = 10
    fixed_inhibition_current = -trial.suggest_float("fixed_inhibition_current", 1e-5, 1e-4)
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=0.0,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=target_activity,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
    )
    network, _, trial = train(
        network,
        dt,
        image_duration,
        n_epochs=2,
        n_samples=n_samples_train,
        log_interval=log_interval,
        import_samples=True,
        trial=trial,
    )
    validation_accuracy = test(
        network,
        dt,
        image_duration=image_duration,
        n_samples=n_samples_validate,
        use_validation_set=True,
        log_interval=log_interval,
        import_samples=True,
    )
    return validation_accuracy