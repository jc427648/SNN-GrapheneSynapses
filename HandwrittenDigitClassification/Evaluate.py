from Network import Network
from Main import train, test
import os
from set_all_seeds import set_all_seeds


def evaluate(n_output_neurons, tau, gamma, n_epochs=1):
    #Following scaling of parameters from previous.
    dt = 2e-4
    image_duration = 0.2
    n_samples_test = 10000
    R = 1000
    v_th_min = 10e-3
    v_th_max = 0.1
    target_activity = 10
    fixed_inhibition_current = -6.0241e-05
    n_samples_train = 60000
    n_samples_test = 10000
    log_interval = 1000
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
    network = train(
        network=network,
        dt=dt,
        image_duration=image_duration,
        n_epochs=n_epochs,
        n_samples=n_samples_train,
        det_training_accuracy=True,
        import_samples=True,
        log_interval=log_interval,
    )[0]
    test_set_accuracy = test(
        network=network,
        dt=dt,
        image_duration=image_duration,
        n_samples=n_samples_test,
        use_validation_set=False,
        import_samples=True,
        log_interval=log_interval,
    )
    return test_set_accuracy
