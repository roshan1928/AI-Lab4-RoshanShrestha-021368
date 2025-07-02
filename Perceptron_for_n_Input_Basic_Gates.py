import numpy as np

from itertools import product

class NInputPerceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.weights = np.zeros(n_inputs)
        self.bias = 0
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        summation = np.dot(x, self.weights) + self.bias
        return self.activation(summation)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
        return self.weights, self.bias

    def evaluate(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy

def generate_truth_table(n):
    return np.array(list(product([0, 1], repeat=n)))

def run_n_input_gate(n, gate_type):
    X = generate_truth_table(n)
    if gate_type == 'AND':
        y = np.array([int(np.all(xi)) for xi in X])
    elif gate_type == 'OR':
        y = np.array([int(np.any(xi)) for xi in X])
    else:
        raise ValueError("Unsupported gate type")

    perceptron = NInputPerceptron(n_inputs=n)
    perceptron.train(X, y)
    acc = perceptron.evaluate(X, y)

    print(f"\n{gate_type} Gate for {n} inputs:")
    print(f"Final Weights: {perceptron.weights}")
    print(f"Final Bias: {perceptron.bias}")
    print(f"Accuracy: {acc * 100:.2f}%")

# Test for 3-input AND/OR
run_n_input_gate(3, 'AND')
run_n_input_gate(3, 'OR')

# Test for 4-input AND/OR
run_n_input_gate(4, 'AND')
run_n_input_gate(4, 'OR')
