import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a threshold gate
def simulate_threshold_gate(input_data, threshold=4):
    outputs = (np.sum(input_data, axis=1) > threshold).astype(int)
    return outputs

# Function to generate a dataset for the threshold gate
def generate_threshold_gate_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 8))  # Random binary inputs
    outputs = simulate_threshold_gate(input_data)  # Simulate the threshold gate outputs

    return input_data, outputs

# Function to create a neural network model for threshold gate emulation
def create_threshold_gate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the threshold gate
num_samples = 10000
input_data, outputs = generate_threshold_gate_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_outputs, test_outputs = train_test_split(
    input_data, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
threshold_gate_model = create_threshold_gate_model()

# Train the model
threshold_gate_model.fit(
    train_input,
    train_outputs,
    epochs=100,
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = threshold_gate_model.evaluate(
    test_input,
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
