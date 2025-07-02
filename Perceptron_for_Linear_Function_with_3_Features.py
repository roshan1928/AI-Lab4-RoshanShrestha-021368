import numpy as np

# Generate 10 samples
np.random.seed(0)
X = np.random.rand(10, 3)  # 10 samples, 3 features
true_weights = np.array([2, 3, 1])
bias = 5
y = X.dot(true_weights) + bias  # Compute target output

# Initialize weights and bias
weights = np.zeros(3)
bias_w = 0
lr = 0.01
epochs = 100

for epoch in range(epochs):
    total_error = 0
    for xi, target in zip(X, y):
        y_pred = np.dot(xi, weights) + bias_w
        error = target - y_pred
        weights += lr * error * xi
        bias_w += lr * error
        total_error += error ** 2
    print(f"Epoch {epoch+1}: Weights = {weights}, Bias = {bias_w:.3f}, MSE = {total_error / len(X):.4f}")

print("\nFinal Learned Weights:", weights)
print("Final Learned Bias:", bias_w)