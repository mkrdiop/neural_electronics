import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
This script generates a dataset for the XOR gate and trains a neural network with one hidden layer. 
The model should achieve good accuracy on the XOR gate dataset. 
Keep in mind that the XOR gate is a classic example where a simple perceptron (single-layer neural network) fails, 
but adding a hidden layer enables the network to learn the XOR function successfully.

"""

# Function to generate a dataset for the XOR gate
def generate_xor_gate_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Input data (0 or 1)
    outputs = np.logical_xor(inputs[:, 0], inputs[:, 1]).astype(int)  # Output is XOR of the inputs

    return inputs, outputs

# Function to create a neural network model for XOR gate
def create_xor_gate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the XOR gate
num_samples = 5000
inputs, outputs = generate_xor_gate_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Create and compile the model
xor_gate_model = create_xor_gate_model()

# Train the model
xor_gate_model.fit(train_inputs, train_outputs, epochs=50, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = xor_gate_model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
