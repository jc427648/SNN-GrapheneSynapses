from Evaluate import evaluate

if __name__ == "__main__":
    n_output_neurons = 500
    tau = 0.074393419260783
    gamma = 0.0076390882373023
    test_set_accuracy = evaluate(n_output_neurons, tau, gamma)
    print(test_set_accuracy)
