import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 1-to-4 Demultiplexer (DEMUX)
def simulate_1_to_4_demux(input_data, control_inputs):
    output1 = np.zeros_like(control_inputs[:, 0])
    output2 = np.zeros_like(control_inputs[:, 0])
    output3 = np.zeros_like(control_inputs[:, 0])
    output4 = np.zeros_like(control_inputs[:, 0])

    for i in range(len(output1)):
        if control_inputs[i, 0] == 0 and control_inputs[i, 1] == 0:
            output1[i] = input_data[i]
        elif control_inputs[i, 0] == 0 and control_inputs[i, 1] == 1:
            output2[i] = input_data[i]
        elif control_inputs[i, 0] == 1 and control_inputs[i, 1] == 0:
            output3[i] = input_data[i]
        elif control_inputs[i, 0] == 1 and control_inputs[i, 1] == 1:
            output4[i] = input_data[i]

    return output1, output2, output3, output4

# Function to generate a dataset for the 1-to-4 DEMUX
def generate_1_to_4_demux_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 1))  # Random binary data input (D)
    control_inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary control inputs (S0, S1)
    output1, output2, output3, output4 = simulate_1_to_4_demux(input_data, control_inputs)  # Simulate the DEMUX outputs

    return input_data, control_inputs, output1, output2, output3, output4

# Function to create a neural network model for 1-to-4 DEMUX emulation
def create_1_to_4_demux_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(4, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(4, activation='sigmoid')  # Output layer with 4 neurons for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 1-to-4 DEMUX
num_samples = 10000  # Increase the dataset size
input_data, control_inputs, output1, output2, output3, output4 = generate_1_to_4_demux_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_control, test_control, train_output1, test_output1, train_output2, test_output2, train_output3, test_output3, train_output4, test_output4 = train_test_split(
    input_data, control_inputs, output1, output2, output3, output4, test_size=0.2, random_state=42
)

# Create and compile the model
demux_model = create_1_to_4_demux_model()

# Train the model
demux_model.fit(
    np.column_stack((train_input, train_control)),  # Concatenate input and control inputs
    np.column_stack((train_output1, train_output2, train_output3, train_output4)),  # Concatenate outputs
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = demux_model.evaluate(
    np.column_stack((test_input, test_control)),  # Concatenate input and control inputs
    np.column_stack((test_output1, test_output2, test_output3, test_output4)),  # Concatenate outputs
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
