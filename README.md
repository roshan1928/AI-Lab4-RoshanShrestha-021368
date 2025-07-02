## 🧠 Artificial Neural Networks: Perceptron

### 1. Perceptron for Linear Function with 3 Features

- **Target Function**: `y = 2x₁ + 3x₂ + x₃ + 5`
- **Features**: 3
- **Learning Rate**: 0.01
- **Output**: Learned weights and bias, training MSE
- **File**: `perceptron_3_features.py`

### 2. Perceptron for Linear Function with n Features

- **Target Function**: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
- **Features**: n (user-defined)
- **Weights**: Randomly initialized
- **Output**: Learned weights, bias, and training error
- **File**: `perceptron_n_features.py`

---

```bash
pip install numpy matplotlib scikit-learn
python perceptron_3_features.py
python perceptron_n_features.py
