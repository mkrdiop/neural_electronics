import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a full adder
def simulate_full_adder(input_data):
    outputs = np.zeros((len(input_data), 2))

    for i in range(len(outputs)):
        A, B, Cin = input_data[i]
        Sum = (A ^ B) ^ Cin  # XOR operation for sum
        Cout = (A & B) | ((A ^ B) & Cin)  # Generate carry-out using AND and OR operations
        outputs[i] = [Sum, Cout]

    return outputs

# Function to generate a dataset for the full adder
def generate_full_adder_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 3))  # Random binary inputs (A, B, Cin)
    outputs = simulate_full_adder(input_data)  # Simulate the full adder outputs

    return input_data, outputs

# Function to create a neural network model for full adder emulation
def create_full_adder_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(4, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(2, activation='sigmoid')  # Output layer with 2 neurons for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the full adder
num_samples = 10000  # Increase the dataset size
input_data, outputs = generate_full_adder_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_outputs, test_outputs = train_test_split(
    input_data, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
full_adder_model = create_full_adder_model()

# Train the model
full_adder_model.fit(
    train_input,
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = full_adder_model.evaluate(
    test_input,
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
