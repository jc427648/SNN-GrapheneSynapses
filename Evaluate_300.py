from cgi import test
from Evaluate import evaluate
import pandas as pd
import numpy as np
import torch
import os
from set_all_seeds import set_all_seeds
import joblib
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
from threadpoolctl import threadpool_limits, threadpool_info
from threadpoolctl import ThreadpoolController

controller = ThreadpoolController()

n_output_neurons = 300
n_epochs = 3
tau = 2.5561286700337
gamma = 2.17794439608506E-08

@wrap_non_picklable_objects
def f(i):
    with threadpool_limits(limits="sequential_blas_under_openmp"):
        torch.set_num_threads(1)
        test_set_accuracy = evaluate(n_output_neurons, tau, gamma, n_epochs=n_epochs)
        print(test_set_accuracy)
        return test_set_accuracy

if __name__ == "__main__":
    set_all_seeds(0)
    test_set_accuraies = Parallel(n_jobs=10)(delayed(f)(i) for i in range(10))
    test_set_accuraies = np.array(test_set_accuraies)
    print(test_set_accuraies)
    mean = test_set_accuraies.mean()
    std = test_set_accuraies.std()
    print(mean)
    print(std)