from Network import Network
from Main import train, test
import os
from Plotting import plotWeights,ReshapeWeights


if __name__ == "__main__":
    n_output_neurons = 100
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 60000
    n_samples_test = 10000
    log_interval = 1000
    R = 1000
    fixed_inhibition_current = -6.02e-3
    gamma = 1e-6
    tau = 3
    n_epochs = 2
    v_th_max = 30
    lower_freq = 20
    upper_freq = 100
    image_threshold = 10
    stdp = 0.1
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=0.0,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=1,
        v_th_min=1e-4,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
        stdp = stdp
    )
    network = train(
        network,
        dt,
        image_duration,
        n_epochs = n_epochs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_train,
        log_interval=log_interval,
        import_samples=True,
    )[0]
    test_set_accuracy = test(
        network,
        dt,
        image_duration=image_duration,
        n_samples=n_samples_test,
        use_validation_set=False,
        log_interval=log_interval,
        import_samples=True,
    )
    print(test_set_accuracy)
