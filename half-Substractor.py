import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a half subtractor
def simulate_half_subtractor(input_data):
    outputs = np.zeros((len(input_data), 2))

    for i in range(len(outputs)):
        A, B = input_data[i]
        D = A ^ B  # XOR operation for difference
        Bout = ~A & B  # Generate borrow using NOT and AND operations
        outputs[i] = [D, Bout]

    return outputs

# Function to generate a dataset for the half subtractor
def generate_half_subtractor_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary inputs (A and B)
    outputs = simulate_half_subtractor(input_data)  # Simulate the half subtractor outputs

    return input_data, outputs

# Function to create a neural network model for half subtractor emulation
def create_half_subtractor_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # Hidden layer with 4 neurons
        tf.keras.layers.Dense(2, activation='sigmoid')  # Output layer with 2 neurons for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the half subtractor
num_samples = 10000  # Increase the dataset size
input_data, outputs = generate_half_subtractor_dataset(num_samples)

# Split the dataset into training and testing sets
train_input, test_input, train_outputs, test_outputs = train_test_split(
    input_data, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
half_subtractor_model = create_half_subtractor_model()

# Train the model
half_subtractor_model.fit(
    train_input,
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = half_subtractor_model.evaluate(
    test_input,
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
