from Network import Network
from Main import train, test
import os


if __name__ == "__main__":
    n_output_neurons = 100
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 60000
    n_samples_test = 10000
    log_interval = 1000
    R = 278.89592139791523
    fixed_inhibition_current = -1.0
    gamma = 0.030239527895736668
    tau = 0.10451865990872834
    v_th_max = 35.79319688022842
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