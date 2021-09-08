from Network import Network
from Main import train, test
import os
from set_all_seeds import set_all_seeds


if __name__ == "__main__":
    set_all_seeds(0)
    n_output_neurons = 10
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 100
    # n_samples_test = 10000
    log_interval = 1
    R = 1000
    fixed_inhibition_current = -1.0
    Ve = 0.0
    gamma = 0.005
    tau = 0.1
    v_th_max = 50
    v_th_min = 0.25
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=Ve,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=10,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
    )
    network, train_set_accuracy = train(
        network,
        dt,
        image_duration,
        n_samples=n_samples_train,
        log_interval=log_interval,
        import_samples=True,
    )
    # test_set_accuracy = test(
    #     network,
    #     dt,
    #     image_duration=image_duration,
    #     n_samples=n_samples_test,
    #     use_validation_set=False,
    #     log_interval=log_interval,
    #     import_samples=True,
    # )
    # print(test_set_accuracy)
