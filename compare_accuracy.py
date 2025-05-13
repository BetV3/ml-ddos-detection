import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from your mini-library (ensure 'code' folder is in your PYTHONPATH or working directory)
from code.network import NeuralNet
from code.layers import Dense
from code.activations import ReLU, Softmax
from code.optimizers import Adam
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

# 1. Prepare dataset
digits = load_digits()
X = digits.data / 16.0      # normalize to [0,1]
y = digits.target
num_classes = len(np.unique(y))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
Y_train = one_hot_encode(y_train, num_classes)
Y_test = one_hot_encode(y_test, num_classes)

# 2. Scratch MLP (using your mini-library)
layers = [
    Dense(X_train.shape[1], 64),
    ReLU(),
    Dense(64, num_classes),
    Softmax()
]

# Create optimizer
optimizer = Adam(learning_rate=0.01)

# Create neural network
scratch_net = NeuralNet(
    layers,
    cross_entropy_loss,
    optimizer
)

print("Training scratch MLP...")
history = scratch_net.train(
    X_train, Y_train,
    batch_size=32,
    epochs=20,
    X_val=X_test, 
    y_val=Y_test
)

# Make predictions
predictions = scratch_net.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
acc_scratch = np.mean(predicted_classes == y_test) * 100
print(f"Scratch MLP Accuracy: {acc_scratch:.2f}%")

# 3. scikit-learn MLPClassifier
sklearn_mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)
print("\nTraining sklearn MLPClassifier...")
sklearn_mlp.fit(X_train, y_train)
acc_sklearn = sklearn_mlp.score(X_test, y_test) * 100
print(f"sklearn MLPClassifier Accuracy: {acc_sklearn:.2f}%")

# 4. Comparison summary
results = pd.DataFrame({
    'Implementation': ['Scratch MLP', 'sklearn MLPClassifier'],
    'Accuracy (%)': [round(acc_scratch, 2), round(acc_sklearn, 2)]
})
print("\nComparison Results:")
print(results.to_string(index=False))
