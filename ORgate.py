import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
This script is similar to the one for the AND gate, 
but it uses the np.max function to compute the output for the OR gate. 
After training, the accuracy should be 100% on the test set.
"""

# Function to generate a dataset for the OR gate
def generate_or_gate_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Input data (0 or 1)
    outputs = np.max(inputs, axis=1)  # Output is the maximum of the inputs (OR gate)

    return inputs, outputs

# Function to create a simple neural network model
def create_or_gate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Increased the number of neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the OR gate
num_samples = 5000
inputs, outputs = generate_or_gate_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Create and compile the model
or_gate_model = create_or_gate_model()

# Train the model
or_gate_model.fit(train_inputs, train_outputs, epochs=50, verbose=1, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = or_gate_model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
