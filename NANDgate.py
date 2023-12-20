import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
"""
This script generates a dataset for the NAND gate and trains a neural network with one hidden layer. 
The model should achieve good accuracy on the NAND gate dataset.
"""


# Function to generate a dataset for the NAND gate
def generate_nand_gate_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Input data (0 or 1)
    outputs = 1 - np.logical_and(inputs[:, 0], inputs[:, 1]).astype(int)  # Output is NAND of the inputs

    return inputs, outputs

# Function to create a neural network model for NAND gate
def create_nand_gate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the NAND gate
num_samples = 5000
inputs, outputs = generate_nand_gate_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Create and compile the model
nand_gate_model = create_nand_gate_model()

# Train the model
nand_gate_model.fit(train_inputs, train_outputs, epochs=50, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = nand_gate_model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
