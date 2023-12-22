import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a T Flip-Flop
def simulate_t_flip_flop(inputs, current_state):
    next_state = np.zeros_like(current_state)

    for i in range(len(inputs)):
        if inputs[i] == 1:  # T=1: Toggle Q
            next_state[i] = 1 - current_state[i]
        else:  # T=0: Maintain the current state
            next_state[i] = current_state[i]

    return next_state

# Function to generate a dataset for the T Flip-Flop
def generate_t_flip_flop_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=num_samples)  # Random binary input (T)
    current_state = np.random.randint(0, 2, size=num_samples)  # Random initial state (Q)
    outputs = simulate_t_flip_flop(inputs, current_state)  # Simulate next state (Q+1)

    return inputs, current_state, outputs

# Function to create a neural network model for T Flip-Flop emulation
def create_t_flip_flop_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),  # Hidden layer with 16 neurons
        tf.keras.layers.Dense(8, activation='relu'),  # Additional hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with a single neuron for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the T Flip-Flop
num_samples = 10000  # Increase the dataset size
inputs, current_state, outputs = generate_t_flip_flop_dataset(num_samples)

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_current_state, test_current_state, train_outputs, test_outputs = train_test_split(
    inputs, current_state, outputs, test_size=0.2, random_state=42
)

# Create and compile the model
t_flip_flop_model = create_t_flip_flop_model()

# Train the model
t_flip_flop_model.fit(
    train_inputs,
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = t_flip_flop_model.evaluate(test_inputs, test_outputs, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
