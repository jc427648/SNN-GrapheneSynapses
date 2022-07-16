from Evaluate import evaluate
import pandas as pd
import numpy as np
import torch
import os
from set_all_seeds import set_all_seeds
import joblib
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


if __name__ == "__main__":
    n_output_neurons = 300
    n_epochs = 3
    tau = 3
    gamma = 1e-6
    set_all_seeds(0)
    def f(i):
        with threadpool_limits(limits=1, user_api='blas'):
            test_set_accuracy = evaluate(n_output_neurons, tau, gamma, n_epochs=n_epochs)
            return test_set_accuracy

    test_set_accuraies = Parallel(n_jobs=10)(delayed(f)(i) for i in range(10))
    test_set_accuraies = np.array(test_set_accuraies)
    print(test_set_accuraies)
    mean = test_set_accuraies.mean()
    std = test_set_accuraies.std()
    print(mean)
    print(std)
