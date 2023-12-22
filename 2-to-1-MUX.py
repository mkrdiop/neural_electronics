import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 2-to-1 Multiplexer (MUX)
def simulate_2_to_1_mux(inputs, control_input):
    outputs = np.zeros_like(control_input)

    for i in range(len(control_input)):
        if control_input[i] == 0:  # If S=0, select D0
            outputs[i] = inputs[i, 0]
        else:  # If S=1, select D1
            outputs[i] = inputs[i, 1]

    return outputs

# Function to generate a dataset for the 2-to-1 MUX
def generate_2_to_1_mux_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary data inputs (D0, D1)
    control_input = np.random.randint(0, 2, size=num_samples)  # Random binary control input (S)
    outputs = simulate_2_to_1_mux(inputs, control_input)  # Simulate the MUX output

    return inputs, control_input, outputs

# Function to create a neural network model for 2-to-1 MUX emulation
def create_2_to_1_mux_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(8, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 2-to-1 MUX
num_samples = 10000  # Increase the dataset size
inputs, control_input, outputs = generate_2_to_1_mux_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_control_input, test_control_input, train_outputs, test_outputs = train_test_split(
    inputs, control_input, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
mux_model = create_2_to_1_mux_model()

# Train the model
mux_model.fit(
    np.column_stack((train_inputs, train_control_input)),  # Concatenate inputs and control input
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = mux_model.evaluate(
    np.column_stack((test_inputs, test_control_input)),  # Concatenate inputs and control input
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
