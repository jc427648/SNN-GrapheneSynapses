# Simple example from https://github.com/fmfn/BayesianOptimization
import bayes_opt
from bayes_opt import BayesianOptimization
from Main import main
import os


def black_box_function(tau, R, gamma, target_activity, v_th_min, v_th_max):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    test_set_accuracy = main(
        n_output_neurons=10,
        n_samples_memory=10,
        dt=0.2e-3,
        image_duration=0.05,
        n_epochs=1,
        lower_freq=20,
        upper_freq=200,
        image_threshold=50,
        n_samples_train=30000,
        n_samples_test=10000,
        Ve=0.0,
        tau=tau,  # Possibly optimise this one
        R=R,  # Possibly optimise this one
        gamma=gamma,  # Optimise this one
        target_activity=target_activity,  # Optimise this one
        v_th_min=v_th_min,  # Optimise this one
        v_th_max=v_th_max,  # Optimise this one
        fixed_inhibition_current=-1.0,
        log_interval=1000,
        det_training_accuracy=True,
    )

    return test_set_accuracy


if __name__ == "__main__":
    # Bounded region of parameter space
    pbounds = {
        "tau": (0.01, 0.5),
        "R": (100, 2000),
        "gamma": (1e-4, 5e-2),
        "target_activity": (1, 100),
        "v_th_min": (0, 5),
        "v_th_max": (6, 15),
    }  # Need to initialise bounds.
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(
        init_points=2,
        n_iter=100,
    )