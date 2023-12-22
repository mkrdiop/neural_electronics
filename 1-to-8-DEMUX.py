import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 1-to-8 Demultiplexer (DEMUX)
def simulate_1_to_8_demux(input_data, control_inputs):
    outputs = np.zeros((len(input_data), 8))

    for i in range(len(outputs)):
        index = control_inputs[i, 0] + 2 * control_inputs[i, 1] + 4 * control_inputs[i, 2]
        outputs[i, index] = input_data[i]

    return outputs

# Function to generate a dataset for the 1-to-8 DEMUX
def generate_1_to_8_demux_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 1))  # Random binary data input (D)
    control_inputs = np.random.randint(0, 2, size=(num_samples, 3))  # Random binary control inputs (S0, S1, S2)
    outputs = simulate_1_to_8_demux(input_data, control_inputs)  # Simulate the DEMUX outputs

    return input_data, control_inputs, outputs

# Function to create a neural network model for 1-to-8 DEMUX emulation
def create_1_to_8_demux_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(8, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(8, activation='sigmoid')  # Output layer with 8 neurons for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 1-to-8 DEMUX
num_samples = 10000  # Increase the dataset size
input_data, control_inputs, outputs = generate_1_to_8_demux_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_control, test_control, train_outputs, test_outputs = train_test_split(
    input_data, control_inputs, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
demux_model = create_1_to_8_demux_model()

# Train the model
demux_model.fit(
    np.column_stack((train_input, train_control)),  # Concatenate input and control inputs
    train_outputs,  # Outputs
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = demux_model.evaluate(
    np.column_stack((test_input, test_control)),  # Concatenate input and control inputs
    test_outputs,  # Outputs
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
