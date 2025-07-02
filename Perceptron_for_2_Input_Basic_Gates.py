import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        summation = np.dot(x, self.weights) + self.bias
        return self.activation(summation)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
            print(f"Weights: {self.weights}, Bias: {self.bias}")

        # Final accuracy
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        print(f"Final Accuracy: {accuracy * 100:.2f}%")
        return self.weights, self.bias

# Inputs and outputs for AND
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

# Inputs and outputs for OR
X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([0, 1, 1, 1])

print("Training for AND Gate:")
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(X_and, y_and)

print("\nTraining for OR Gate:")
perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_or, y_or)
