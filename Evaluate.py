from Network import Network
from MNISTDataLoader import getMNIST
from Main import train, test
import os


def evaluate(n_output_neurons, tau, gamma, n_epochs=1):
    dt = 0.2e-3
    Ve = 0.0
    image_duration = 0.05
    R = 1000
    v_th_min = 0.0001
    v_th_max = 30
    fixed_inhibition_current = -0.00602
    image_threshold = 10
    lower_freq = 20
    upper_freq = 100
    target_activity = 1
    # n_samples_train = 60000
    n_samples_train = 10000
    # n_samples_test = 10000
    n_samples_test = 1000
    C2CD2D = 0.5
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=Ve,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=target_activity,
        v_th_min=v_th_min,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
        C2CD2D=C2CD2D,
    )
    ((train_data, train_labels), _, (test_data, test_labels)) = getMNIST(
        load_train_samples=True,
        load_validation_samples=False,
        load_test_samples=True,
        export_to_disk=False,
    )
    network, _ = train(
        network=network,
        dt=dt,
        image_duration=image_duration,
        n_epochs=n_epochs,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_train,
        det_training_accuracy=False,
        data=train_data,
        labels=train_labels,
        trial=None
    )
    test_set_accuracy = test(
        network=network,
        dt=dt,
        image_duration=image_duration,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
        n_samples=n_samples_test,
        data=test_data,
        labels=test_labels,
    )
    return test_set_accuracy

# if __name__ == "__main__":
print("------------")
n_output_neurons = 10
n_epochs = 1
tau = 4.8025530126437
gamma = 0.00002660854454112
test_set_accuracy = evaluate(n_output_neurons=n_output_neurons, tau=tau, gamma=gamma, n_epochs=n_epochs)
print(test_set_accuracy)