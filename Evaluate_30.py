from Network import Network
from Main import train, test
import os


if __name__ == "__main__":
    n_output_neurons = 30
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 60000
    n_samples_test = 10000
    log_interval = 1000
    R = 999.3670475732349
    fixed_inhibition_current = -0.8506349410078146
    gamma = 0.01889405435302123
    tau = 0.1249809832553554
    v_th_max = 49.903031695016146
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
        dt,
        image_duration=image_duration,
        n_samples=n_samples_test,
        use_validation_set=False,
        log_interval=log_interval,
    )
    print(validation_accuracy)