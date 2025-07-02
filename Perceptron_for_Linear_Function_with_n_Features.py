import numpy as np

# Settings
n = 5  # number of features
samples = 10
np.random.seed(42)

# Generate random data
X = np.random.rand(samples, n)
true_weights = np.random.uniform(-1, 1, n)
true_bias = 5
y = X.dot(true_weights) + true_bias  # Linear function with noise-free data

# Initialize model weights
weights = np.zeros(n)
bias_w = 0
lr = 0.01
epochs = 100

# Training loop
for epoch in range(epochs):
    total_error = 0
    for xi, target in zip(X, y):
        y_pred = np.dot(xi, weights) + bias_w
        error = target - y_pred
        weights += lr * error * xi
        bias_w += lr * error
        total_error += error ** 2
    print(f"Epoch {epoch+1}: Weights = {weights}, Bias = {bias_w:.3f}, MSE = {total_error / samples:.4f}")

print("\nFinal Learned Weights:", weights)
print("Final Learned Bias:", bias_w)