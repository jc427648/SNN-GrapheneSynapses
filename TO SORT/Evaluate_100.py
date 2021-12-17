from Evaluate import evaluate

if __name__ == "__main__":
    n_output_neurons = 100
    tau = 0.0808634308714319
    gamma = 0.00779422611516095
    test_set_accuracy = evaluate(n_output_neurons, tau, gamma)
    print(test_set_accuracy)
