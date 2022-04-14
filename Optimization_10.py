# https://optuna.org/
from Network import Network
from Main import train, test
import os
import joblib
import optuna
from optuna.trial import TrialState
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils



def objective(trial, n_output_neurons):
    Ve = 0.0
    tau = trial.suggest_float("tau", 1e-3, 1e-1)  
    gamma = trial.suggest_float("gamma", 1e-7, 1e-5)
    R = 500
    v_th_min = 0.001
    v_th_max = 30
    fixed_inhibition_current = -0.00602
    dt = 0.2e-3
    image_duration = 0.05
    image_threshold = 50
    lower_freq = 20
    upper_freq = 100
    n_samples_train = 60000
    n_samples_test = 10000
    n_epochs = 1
  
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
        import_samples=False,
    )
    test_set_accuracy = test(
        network=network,
        dt=dt,
        image_duration=image_duration,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_test,
        use_validation_set=False,
        import_samples=False,
    )
    return test_set_accuracy

class RepeatPruner(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        trials = study.get_trials(deepcopy=False)
        completed_trials = [t.params for t in trials if t.state == TrialState.COMPLETE]
        n_trials = len(completed_trials)
        if n_trials == 0:
            return False

        if trial.params in completed_trials:
            return True

        return False

if __name__ == "__main__":
    n_output_neurons = 10
    n_trials = 100
    sampler = optuna.samplers.TPESampler()
    run = neptune.init(api_token=os.getenv("NEPTUNE_API_TOKEN"),
                       project='JCU-NICE/Updated-SNN-Optimization',
                       mode="offline")
    neptune_callback = optuna_utils.NeptuneCallback(run)
    storage_name = "sqlite:///{}.db".format(n_output_neurons)
    study = optuna.create_study(study_name=str(n_output_neurons), direction="maximize", sampler=sampler, storage=storage_name, load_if_exists=True, pruner=RepeatPruner())
    # study.enqueue_trial({'tau': 0.01,
    #                      'gamma': 0.000001,
    #                     })
    study.optimize(lambda trial: objective(
        trial, n_output_neurons), n_trials=n_trials, callbacks=[neptune_callback])
    run.stop()
