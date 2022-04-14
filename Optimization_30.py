# https://optuna.org/
from Network import Network
from MNISTDataLoader import getMNIST
from Main import train, test
import os
import joblib
import optuna
from optuna.trial import TrialState
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils


def objective(trial, n_output_neurons, n_epochs):
    Ve = 0.0
    tau = trial.suggest_float("tau", 1e-5, 5e-1, log=True)  
    gamma = trial.suggest_float("gamma", 1e-8, 5e-4, log=True)
    R = 500
    v_th_min = 0.001
    v_th_max = 30
    fixed_inhibition_current = -0.00602
    dt = 0.2e-3
    image_duration = 0.05
    image_threshold = 50
    lower_freq = 20
    upper_freq = 100
    n_samples_train = 50000
    n_samples_validate = 10000
  
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=Ve,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=n_output_neurons,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
    )
    ((train_data, train_labels), (validation_data, validation_labels), _) = getMNIST(
        load_train_samples=True,
        load_validation_samples=True,
        load_test_samples=False,
        validation_samples=n_samples_validate,
        export_to_disk=False,
    )

    network, _ = train(
        network=network,
        dt=dt,
        image_duration=image_duration,
        n_epochs=n_epochs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_train,
        det_training_accuracy=False,
        data=train_data,
        labels=train_labels,
        trial=trial,
    )
    validation_set_accuracy = test(
        network=network,
        dt=dt,
        image_duration=image_duration,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_validate,
        data=validation_data,
        labels=validation_labels,
    )
    return validation_set_accuracy


if __name__ == "__main__":
    n_output_neurons = 30
    n_epochs = 1
    n_trials = 10000
    sampler = optuna.samplers.TPESampler()
    run = neptune.init(api_token=os.getenv("NEPTUNE_API_TOKEN"),
                       project='JCU-NICE/Updated-SNN-Optimization',
                       mode="offline")
    neptune_callback = optuna_utils.NeptuneCallback(run)
    storage_name = "sqlite:///{}.db".format(n_output_neurons)
    study = optuna.create_study(study_name=str(n_output_neurons), direction="maximize", sampler=sampler, storage=storage_name, load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    # study.enqueue_trial({'tau': 0.0592626094478231,
    #                      'gamma': 5.72033207422828E-06,
    #                     })
    study.optimize(lambda trial: objective(
        trial, n_output_neurons, n_epochs), n_trials=n_trials, callbacks=[neptune_callback])
    run.stop()
