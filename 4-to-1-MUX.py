import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 4-to-1 Multiplexer (MUX)
def simulate_4_to_1_mux(inputs, control_inputs):
    outputs = np.zeros_like(control_inputs[:, 0])

    for i in range(len(outputs)):
        control_bits = control_inputs[i, :]
        selected_input = np.argmax(control_bits)  # Convert control bits to decimal to select the input
        outputs[i] = inputs[i, selected_input]

    return outputs

# Function to generate a dataset for the 4-to-1 MUX
def generate_4_to_1_mux_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 4))  # Random binary data inputs (D0, D1, D2, D3)
    control_inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary control inputs (S0, S1)
    outputs = simulate_4_to_1_mux(inputs, control_inputs)  # Simulate the MUX output

    return inputs, control_inputs, outputs

# Function to create a neural network model for 4-to-1 MUX emulation
def create_4_to_1_mux_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(8, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 4-to-1 MUX
num_samples = 10000  # Increase the dataset size
inputs, control_inputs, outputs = generate_4_to_1_mux_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_control_inputs, test_control_inputs, train_outputs, test_outputs = train_test_split(
    inputs, control_inputs, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
mux_model = create_4_to_1_mux_model()

# Train the model
mux_model.fit(
    np.column_stack((train_inputs, train_control_inputs)),  # Concatenate inputs and control inputs
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = mux_model.evaluate(
    np.column_stack((test_inputs, test_control_inputs)),  # Concatenate inputs and control inputs
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
