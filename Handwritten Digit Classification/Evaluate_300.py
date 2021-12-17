from Evaluate import evaluate
import pandas as pd
import numpy as np
import torch
import os
from Mailgun import send_email
from set_all_seeds import set_all_seeds
import joblib
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


if __name__ == "__main__":
    n_output_neurons = 500
    tau = 0.0845993883511427
    gamma = 0.00993280807733143
    set_all_seeds(0)

    def f(i):
        with threadpool_limits(limits=1, user_api='blas'):
            test_set_accuracy = evaluate(n_output_neurons, tau, gamma, n_epochs=3)
            print(test_set_accuracy)
            return test_set_accuracy

    test_set_accuraies = Parallel(n_jobs=10)(delayed(f)(i) for i in range(10))
    test_set_accuraies = np.array(test_set_accuraies)
    print(test_set_accuraies)
    mean = test_set_accuraies.mean()
    std = test_set_accuraies.std()
    print(mean)
    print(std)
    send_email('Awoogna', 'Evaluate_500 (5) Executed', 'mean: %f, std: %f' % (mean, std))
