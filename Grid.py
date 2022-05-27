import itertools
import os
import pandas as pd
import argparse

from Network import Network
from MNISTDataLoader import getMNIST
from Main import train, test


def run(
    n_output_neurons,
    Ve,
    tau,
    R,
    gamma,
    target_activity,
    v_th_min,
    v_th_max,
    fixed_inhibition_current,
    dt,
    image_duration,
    image_threshold,
    lower_freq,
    upper_freq,
    n_samples_train,
    n_samples_test,
    n_epochs,
    C2CD2D,
    UUID,
):
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
    df = pd.read_csv(os.path.join(os.getcwd(), "grid_out.csv"))
    df = df.append(
        {
            "n_output_neurons": n_output_neurons,
            "Ve": Ve,
            "tau": tau,
            "R": R,
            "gamma": gamma,
            "target_activity": target_activity,
            "v_th_min": v_th_min,
            "v_th_max": v_th_max,
            "fixed_inhibition_current": fixed_inhibition_current,
            "dt": dt,
            "image_duration": image_duration,
            "image_threshold": image_threshold,
            "lower_freq": lower_freq,
            "upper_freq": upper_freq,
            "n_epochs": n_epochs,
            "test_set_accuracy": test_set_accuracy,
            "C2CD2D": C2CD2D,
            "UUID": UUID,
        },
        ignore_index=True,
    )
    df.to_csv(os.path.join(os.getcwd(), "grid_out.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_output_neurons", type=int, default=100)
    parser.add_argument("--Ve", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=3e-0)
    parser.add_argument("--R", type=float, default=1000)
    parser.add_argument("--gamma", type=float, default=1e-6)
    parser.add_argument("--target_activity", type=float, default=1)
    parser.add_argument("--v_th_min", type=float, default=0.0001)
    parser.add_argument("--v_th_max", type=float, default=30)
    parser.add_argument("--fixed_inhibition_current", type=float, default=-0.00602)
    parser.add_argument("--dt", type=float, default=0.2e-3)
    parser.add_argument("--image_duration", type=float, default=0.05)
    parser.add_argument("--image_threshold", type=float, default=10)
    parser.add_argument("--lower_freq", type=float, default=20)
    parser.add_argument("--upper_freq", type=float, default=100)
    parser.add_argument("--n_samples_train", type=int, default=60000)
    parser.add_argument("--n_samples_test", type=int, default=10000)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--C2CD2D", type = float, default=0.)
    parser.add_argument("--UUID", type=str, default=None)
    args = parser.parse_args()
    if os.path.exists(os.path.join(os.getcwd(), "grid_out.csv")):
        df = pd.read_csv(os.path.join(os.getcwd(), "grid_out.csv"))
    else:
        df = pd.DataFrame(
            columns=[
                "n_output_neurons",
                "Ve",
                "tau",
                "R",
                "gamma",
                "target_activity",
                "v_th_min",
                "v_th_max",
                "fixed_inhibition_current",
                "dt",
                "image_duration",
                "image_threshold",
                "lower_freq",
                "upper_freq",
                "n_epochs",
                "test_set_accuracy",
                "C2CD2D",
                "UUID",
            ]
        )
        df.to_csv(os.path.join(os.getcwd(), "grid_out.csv"), index=False)

    run(**vars(args))