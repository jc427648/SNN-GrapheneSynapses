# https://github.com/fmfn/BayesianOptimization
import bayes_opt
from bayes_opt import BayesianOptimization
from Network import Network
from Main import train, test
import os


def black_box_function(tau, R, gamma, v_th_max, fixed_inhibition_current):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    n_output_neurons = 100
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 20000
    n_samples_validate = 5000
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
        n_output_neurons,
        dt,
        image_duration,
        n_samples=n_samples_train,
        log_interval=log_interval,
    )[0]
    validation_accuracy = test(
        network,
        n_output_neurons,
        dt,
        image_duration=image_duration,
        n_samples=n_samples_validate,
        use_validation_set=True,
        log_interval=log_interval,
    )
    return validation_accuracy


if __name__ == "__main__":
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