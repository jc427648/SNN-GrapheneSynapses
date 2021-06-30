# Test parameters from BayesianOptimization_10.py (iteration 33)
from Network import Network
from Maintmp import train, test
import os


if __name__ == "__main__":
    n_output_neurons = 10
    n_samples_memory = 10
    n_epochs = 1
    Ve = 0.0
    tau = 0.4218
    R = 518.3
    gamma = 0.02802
    target_activity = 21.07
    v_th_min = 0.5578
    v_th_max = 8.709
    fixed_inhibition_current = -1.0
    dt = 0.2e-3
    output_dir = "output"
    n_samples_train = 60000
    n_samples_test = 10000
    image_duration = 0.05
    lower_freq = 20
    upper_freq = 200
    image_threshold = 50
    log_interval = 1000
    det_training_accuracy = True

    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_samples_memory,
        Ve=Ve,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=target_activity,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
        output_dir=output_dir,
    )
    network, training_set_accuracy = train(
        network,
        n_samples=n_samples_train,
        dt=dt,
        image_duration=image_duration,
        n_epochs=n_epochs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        log_interval=log_interval,
        det_training_accuracy=det_training_accuracy,
    )
    test_set_accuracy = test(
        network,
        n_samples=n_samples_test,
        dt=dt,
        image_duration=image_duration,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        log_interval=log_interval,
    )
    print(test_set_accuracy)