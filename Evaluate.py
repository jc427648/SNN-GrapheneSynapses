from Network import Network
from Main import train, test
import os
from Plotting import plotWeights,ReshapeWeights


if __name__ == "__main__":
    n_output_neurons = 10
    dt = 0.2e-3
    image_duration = 0.05
    n_samples_train = 60000
    n_samples_test = 10000
    log_interval = 1000
    R = 1000
    fixed_inhibition_current = -6.02e-5
    gamma = 5e-4
    tau = 0.1e-2
    v_th_max = 30
    PlotTitle = 'GamOriginal'
    lower_freq = 5
    upper_freq = 200
    image_threshold = 200
    network = Network(
        n_output_neurons=n_output_neurons,
        n_samples_memory=n_output_neurons,
        Ve=0.0,
        tau=tau,
        R=R,
        gamma=gamma,
        target_activity=10,
        v_th_min=10e-3,
        v_th_max=v_th_max,
        fixed_inhibition_current=fixed_inhibition_current,
        dt=dt,
    )
    network = train(
        network,
        dt,
        image_duration,
        lower_freq=lower_freq,
        upper_freq=upper_freq,
        image_threshold=image_threshold,
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

    string = 'Wmax = %f, Tau = %f, Gamma = %f, R = %f, Target Activity = %f, Image Threshold = %f' %(
        network.synapse.wmax,
        network.group.tau,
        network.group.gamma,
        network.group.R,
        network.group.target,
        image_threshold
    )

    print(string)


    plotStringWeights = string + 'Weights'
    plotStringConfusion = string + 'Confusion'
    #Plot, save and store weights.
    RWeights,assignments = ReshapeWeights(network.synapse.w,network.n_output_neurons)
    plotWeights(RWeights,network.synapse.wmax,network.synapse.wmin,title = PlotTitle)

    # torch.save(network.Assignment,'Assignments.pt')
    # torch.save(network.Activity,'Activity.pt')
    print('Assignment:')
    print('\n')
    print(network.Assignment)
    print('\n')
    print('Activity:')
    print('\n')
    print(network.Activity)
    print('\n')
    print('Vth:')
    print('\n')
    print(network.group.Vth)