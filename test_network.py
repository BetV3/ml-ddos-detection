import sys
import os
import numpy as np

# Add the parent directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.layers import Dense
from code.activations import ReLU, Softmax
from code.network import NeuralNet
from code.optimizers import SGD
from code.losses import cross_entropy_loss
from code.utils import one_hot_encode

# Generate some random data
np.random.seed(42)
X = np.random.randn(100, 10)
y = np.random.randint(0, 3, size=100)
y_one_hot = one_hot_encode(y, 3)

# Create a simple neural network
layers = [
    Dense(10, 20),
    ReLU(),
    Dense(20, 3),
    Softmax()
]

# Create optimizer and network
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model = NeuralNet(layers, cross_entropy_loss, optimizer)

# Train the model for a few epochs
history = model.train(X, y_one_hot, batch_size=32, epochs=5)

# Make predictions
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == y)
print(f"Accuracy: {accuracy:.4f}") 