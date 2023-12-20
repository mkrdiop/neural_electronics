import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a JK Flip-Flop
def simulate_jk_flip_flop(inputs, current_state):
    next_state = np.zeros_like(current_state)

    for i in range(len(inputs)):
        if inputs[i, 0] == 1 and inputs[i, 1] == 0:  # J=1, K=0: Set Q to 1
            next_state[i] = 1
        elif inputs[i, 0] == 0 and inputs[i, 1] == 1:  # J=0, K=1: Reset Q to 0
            next_state[i] = 0
        elif inputs[i, 0] == 1 and inputs[i, 1] == 1:  # J=1, K=1: Toggle Q
            next_state[i] = 1 - current_state[i]
        else:  # J=0, K=0: Maintain the current state
            next_state[i] = current_state[i]

    return next_state

# Function to generate a dataset for the JK Flip-Flop
def generate_jk_flip_flop_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary inputs (J, K)
    current_state = np.random.randint(0, 2, size=num_samples)  # Random initial state (Q)
    outputs = simulate_jk_flip_flop(inputs, current_state)  # Simulate next state (Q+1)

    return inputs, current_state, outputs

# Function to create a neural network model for JK Flip-Flop emulation
def create_jk_flip_flop_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(8, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the JK Flip-Flop
num_samples = 10000  # Increase the dataset size
inputs, current_state, outputs = generate_jk_flip_flop_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_current_state, test_current_state, train_outputs, test_outputs = train_test_split(
    inputs, current_state, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
jk_flip_flop_model = create_jk_flip_flop_model()

# Train the model
jk_flip_flop_model.fit(
    train_inputs,
    train_outputs,
    epochs=200,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = jk_flip_flop_model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
