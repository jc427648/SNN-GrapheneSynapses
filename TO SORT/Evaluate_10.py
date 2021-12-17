from Evaluate import evaluate

if __name__ == "__main__":
    n_output_neurons = 10
    tau = 0.14986746754094
    gamma = 0.00630155728720963
    test_set_accuracy = evaluate(n_output_neurons, tau, gamma)
    print(test_set_accuracy)
