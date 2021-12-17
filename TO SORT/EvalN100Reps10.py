from Network import Network
from Main import train, test
import os


if __name__ == "__main__":
    for i in range(10):
        n_output_neurons = 100
        dt = 0.2e-3
        image_duration = 0.05
        n_epochs = 1
        n_samples_train = 60000
        n_samples_test = 10000
        log_interval = 1000
        R = 1000
        fixed_inhibition_current = -1.0
        gamma = 0.005
        tau = 0.1
        v_th_max = 50
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

        string = 'Parameters used: No. of Neurons = %f, R = %f, Tau = %f, VthMax = %f, Target = %f, F.I.C = %f, gamma = %f, VthMin = %f, Ve = %f' %(

            network.n_output_neurons,
            network.group.R,
            network.group.tau,
            network.group.VthMax,
            network.group.target,
            network.fixed_inhibition_current,
            network.group.gamma,
            network.group.VthMin,
            network.group.Ve
        )

        print(string)