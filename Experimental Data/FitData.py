from Model import VTEAMMod, RMSErr
import numpy as np
import optuna
import matplotlib.pyplot as plt
from set_all_seeds import set_all_seeds
from threadpoolctl import threadpool_limits


n_trials = 5000
# r_sq_prune_thresholds = [0.8, 0.85, 0.9, 0.95, 0.975]  # Prune trials with a (near) linear I/V relationship
r_sq_prune_thresholds = [0.945, 0.94]
n_cpus = 4  # Number of CPUs to use
top_n = 50  # Number of top results to be plotted

with threadpool_limits(limits=n_cpus, user_api='blas'):
    for r_sq_prune_threshold in r_sq_prune_thresholds:
        f1 = np.loadtxt("pentacenesingle200_4_slow_k2400.txt", skiprows=1)
        f2 = np.loadtxt("pentacenesingle200_3_slow_k2400.txt", skiprows=1)
        I_ref1 = f1[:, 3]
        V_ref1 = f1[:, 2]
        I_ref2 = f2[:, 3]
        V_ref2 = f2[:, 2]
        # Remove the anomalous spike from pentacenesingle200_4_slow_k2400
        I_ref1[6864] = I_ref1[6863]
        I_ref1[6865] = I_ref1[6863]
        # Average I/V data
        I_ref_avg = (I_ref1 + I_ref2) / 2

        set_all_seeds(0)
        study = optuna.create_study(direction="minimize")
        study.enqueue_trial(
            {
                "alphaSetp": -1,
                "alphaSetn": 2.97,
                "alphaResetp": 1.01,
                "alphaResetn": 1.01,
                "kSetp": 0.00185,
                "kSetn": 1,
                "VResetp": 0.368,
                "VResetn": -0.1,
                "ROff": 20695,
                "ROn": 1550,
                "VSetp": 1.1,
                "VSetn": -1.35,
                "kResetp": -26,
                "kResetn": -26,
            }
        )
        study.optimize(
            lambda trial: RMSErr(trial, I_ref_avg, V_ref1, r_sq_prune_threshold),
            n_trials=n_trials,
            n_jobs=n_cpus,
        )

        print("Best value: ", study.best_trial.value)
        study_df = study.trials_dataframe().sort_values(by=["value"], ascending=True)
        study_df.dropna(subset=["value"], inplace=True)
        study_df = study_df.iloc[: min(top_n, len(study_df))]
        study_df.to_csv("optimization_%0.3f.csv" % r_sq_prune_threshold, index=False)

        plt.figure(1)
        plt.clf()
        plt.plot(V_ref1, I_ref_avg, "k", label="Reference")
        for i in range(len(study_df)):
            row_df = study_df.iloc[i, :].to_dict()
            trial_idx = row_df["number"]
            optimized_params = {}
            for key in row_df.keys():
                if key.startswith("params_"):
                    optimized_params[key[len("params_") :]] = row_df[key]

            model = VTEAMMod(0, **optimized_params)
            plt.plot(V_ref1, model.GetCurrentOutput(V_ref1, 0.2), label="Trial %d" % trial_idx)

        plt.legend()
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.savefig("fit_%0.3f.svg" % r_sq_prune_threshold)
        # plt.show()
