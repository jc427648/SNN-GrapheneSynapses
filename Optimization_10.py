# https://optuna.org/
from Network import Network
from Main import train, test
import os
import optuna
from optuna.trial import TrialState

# •	n_samples_memory = 100
# •	Ve = 0.0
# •	tau = 0.002
# •	R = 20
# •	gamma = 0.0005
# •	target activity = 10
# •	VthMin = 0.01
# •	VthMax = 1
# •	fixed inhibition current = -0.85

def objective(trial):
    """ Function with unknown internals we wish to maximize.
    """

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)



    n_output_neurons = 30
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 50000
    n_samples_validate = 10000
    log_interval = 1000
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=0.0,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=10,
        v_th_min=0.25,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
    )
    network = train(
        network,
        dt,
        image_duration,
        n_samples=n_samples_train,
        log_interval=log_interval,
        import_samples=True,
    )[0]
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


if __name__ == "__main__":
    # Parameters
    n_output_neurons = 10




    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])













    
    # Bounded region of parameter space
    pbounds = {
        "tau": (0.01, 0.2),
        "R": (100, 2000),
        "gamma": (1e-4, 5e-2),
        "v_th_max": (1, 50),
        "fixed_inhibition_current": (-1.0, -0.1),
    }  # Need to initialise bounds.
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.probe(
        params={
            "tau": 0.1,
            "R": 1000,
            "gamma": 0.005,
            "v_th_max": 50,
            "fixed_inhibition_current": -1.0,
        }
    )
    optimizer.maximize(
        init_points=5,
        n_iter=94,
    )
    print(optimizer.max)