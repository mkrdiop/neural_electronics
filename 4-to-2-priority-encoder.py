import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 4-to-2 priority encoder
def simulate_4_to_2_priority_encoder(input_data):
    outputs = np.zeros((len(input_data), 2))

    for i in range(len(outputs)):
        A, B, C, D = input_data[i]
        if A == 1:
            outputs[i] = [0, 0]
        elif B == 1:
            outputs[i] = [0, 1]
        elif C == 1:
            outputs[i] = [1, 0]
        elif D == 1:
            outputs[i] = [1, 1]

    return outputs

# Function to generate a dataset for the 4-to-2 priority encoder
def generate_4_to_2_priority_encoder_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 4))  # Random binary inputs (A, B, C, D)
    outputs = simulate_4_to_2_priority_encoder(input_data)  # Simulate the 4-to-2 priority encoder outputs

    return input_data, outputs

# Function to create a neural network model for 4-to-2 priority encoder emulation
def create_4_to_2_priority_encoder_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(2, activation='softmax')  # Output layer with 2 neurons for multi-class classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 4-to-2 priority encoder
num_samples = 10000  # Increase the dataset size
input_data, outputs = generate_4_to_2_priority_encoder_dataset(num_samples)

# One-hot encode the output labels
outputs_onehot = tf.keras.utils.to_categorical(outputs.argmax(axis=1), num_classes=2)

# Split the dataset into training and testing sets
train_input, test_input, train_outputs, test_outputs = train_test_split(
    input_data, outputs_onehot, test_size=0.2, random_state=42
)

# Create and compile the model
priority_encoder_4_to_2_model = create_4_to_2_priority_encoder_model()

# Train the model
priority_encoder_4_to_2_model.fit(
    train_input,
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = priority_encoder_4_to_2_model.evaluate(
    test_input,
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
