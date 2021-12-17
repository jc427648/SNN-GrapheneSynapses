from Evaluate import evaluate

if __name__ == "__main__":
    n_output_neurons = 30
    tau = 0.0588411280556185
    gamma = 0.00569002917217075
    test_set_accuracy = evaluate(n_output_neurons, tau, gamma)
    print(test_set_accuracy)
