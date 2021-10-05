from Evaluate import evaluate

if __name__ == "__main__":
    n_output_neurons = 300
    tau = 0.0665461362061757
    gamma = 0.00960571567431384
    test_set_accuracy = evaluate(n_output_neurons, tau, gamma)
    print(test_set_accuracy)
