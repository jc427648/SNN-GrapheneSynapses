from cgi import test
from Evaluate import evaluate
import pandas as pd
import numpy as np
import torch
import os
from set_all_seeds import set_all_seeds
import joblib
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from Plotting import ReshapeWeights,plotWeights,plot_confusion_matrix
from Network import Network
from STDPsynapses import LIFNeuronGroup,STDPSynapse


if __name__ == "__main__":
    #Following scaling of parameters like previous
    n_output_neurons = 10
    n_epochs = 1
    tau = 0.0015744
    gamma = 0.005e-2#This is a new parameter. You need to check Corey's again.
    set_all_seeds(0)
    
    # def f(i):
        # with threadpool_limits(limits=1, user_api='blas'):
    test_set_accuracy,network = evaluate(n_output_neurons, tau, gamma, n_epochs=n_epochs)
    print(test_set_accuracy)
    print('\n')

    string = 'Wmax = %f, Tau = %f, Gamma = %f, R = %f, Target Activity = %f' %(
        network.synapse.wmax,
        network.group.tau,
        network.group.gamma,
        network.group.R,
        network.group.target
    )

    print(string)


    plotStringWeights = string + 'Weights'
    plotStringConfusion = string + 'Confusion'
    #Plot, save and store weights.
    RWeights,assignments = ReshapeWeights(network.synapse.w,network.n_output_neurons)
    plotWeights(RWeights,network.synapse.wmax,network.synapse.wmin,title = 'NormalRun')

    torch.save(network.Assignment,'Assignments.pt')
    torch.save(network.Activity,'Activity.pt')

    print(network.Assignment)
    print(network.Activity)
    #Ignoring confusion matrix at this stage.
    #plot_confusion_matrix(assign)





    # return test_set_accuracy

    # test_set_accuraies = Parallel(n_jobs=10)(delayed(f)(i) for i in range(10))
    # test_set_accuraies = np.array(test_set_accuraies)
    # print(test_set_accuraies)
    # mean = test_set_accuraies.mean()
    # std = test_set_accuraies.std()
    # print(mean)
    # print(std)