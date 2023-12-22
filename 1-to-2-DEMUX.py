import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 1-to-2 Demultiplexer (DEMUX)
def simulate_1_to_2_demux(input_data, control_input):
    output1 = np.zeros_like(control_input)
    output2 = np.zeros_like(control_input)

    for i in range(len(control_input)):
        if control_input[i] == 0:
            output1[i] = input_data[i]
        else:
            output2[i] = input_data[i]

    return output1, output2

# Function to generate a dataset for the 1-to-2 DEMUX
def generate_1_to_2_demux_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 1))  # Random binary data input (D)
    control_input = np.random.randint(0, 2, size=(num_samples, 1))  # Random binary control input (S)
    output1, output2 = simulate_1_to_2_demux(input_data, control_input)  # Simulate the DEMUX outputs

    return input_data, control_input, output1, output2

# Function to create a neural network model for 1-to-2 DEMUX emulation
def create_1_to_2_demux_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(4, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(2, activation='sigmoid')  # Output layer with 2 neurons for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 1-to-2 DEMUX
num_samples = 10000  # Increase the dataset size
input_data, control_input, output1, output2 = generate_1_to_2_demux_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_control, test_control, train_output1, test_output1, train_output2, test_output2 = train_test_split(
    input_data, control_input, output1, output2, test_size=0.2, random_state=42
)

# Create and compile the model
demux_model = create_1_to_2_demux_model()

# Train the model
demux_model.fit(
    np.column_stack((train_input, train_control)),  # Concatenate input and control input
    np.column_stack((train_output1, train_output2)),  # Concatenate output1 and output2
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = demux_model.evaluate(
    np.column_stack((test_input, test_control)),  # Concatenate input and control input
    np.column_stack((test_output1, test_output2)),  # Concatenate output1 and output2
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
