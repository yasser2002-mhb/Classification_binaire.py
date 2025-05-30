import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Neural network class
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = relu(Z)
            self.activations.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        A = sigmoid(Z)
        self.activations.append(A)
        return A

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-8  # Avoid log(0)
        return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

    def backward(self, y_true):
        m = y_true.shape[0]
        dA = self.activations[-1] - y_true
        self.d_weights = []
        self.d_biases = []

        for i in reversed(range(len(self.weights))):
            dZ = dA
            A_prev = self.activations[i]
            dW = (A_prev.T @ dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.d_weights.insert(0, dW)
            self.d_biases.insert(0, db)

            if i > 0:
                dA = (dZ @ self.weights[i].T) * relu_derivative(self.z_values[i-1])

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32):
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X = X[indices]
            y = y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                outputs = self.forward(X_batch)
                self.backward(y_batch)
                self.update_weights()
            # Compute loss
            train_loss = self.compute_loss(y, self.forward(X))
            val_loss = self.compute_loss(y_val, self.forward(X_val))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        return train_losses, val_losses

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)

# Load and preprocess data
df = pd.read_csv("diabetes.csv")

# Replace zeros with median in relevant columns
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values.reshape(-1, 1)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Model configuration
nn = NeuralNetwork(layer_sizes=[8, 16, 8, 1], learning_rate=0.01)
train_losses, val_losses = nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Evaluate
y_pred = nn.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Loss Curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
