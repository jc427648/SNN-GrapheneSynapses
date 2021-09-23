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
                       project='JCU-NICE/Final-SNN-Optimization')
    neptune_callback = optuna_utils.NeptuneCallback(run)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({'tau': 0.05, 'gamma': 1e-3,
                        'target_activity': 10, 'v_th_max': 50, 'fixed_inhibition_current': -1.0})
    study.optimize(lambda trial: objective(
        trial, n_output_neurons), n_trials=n_trials, callbacks=[neptune_callback])
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])
    run.stop()


def objective(trial, n_output_neurons):
    """ Function with unknown internals we wish to maximize.
    """
    tau = trial.suggest_float("tau", 0.05, 0.15, log=True)
    gamma = trial.suggest_float("gamma", 1e-3, 1e-2, log=True)
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 50000
    n_samples_validate = 10000
    log_interval = 5000
    R = 1000
    v_th_max = 50
    v_th_min = 0.25
    target_activity = 10
    fixed_inhibition_current = -1.0
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
