import tensorflow as tf
import numpy as np

# Function to simulate JK Flip-Flop behavior
def simulate_jk_flip_flop(inputs, current_state):
    # JK Flip-Flop truth table logic
    j = inputs[:, 0]
    k = inputs[:, 1]

    # Determine the next state
    next_state = (current_state & ~j) | (j & ~k)

    return next_state

# Generate training dataset for JK Flip-Flop
def generate_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 2))  # Input data (J, K)
    current_state = np.random.randint(0, 2, size=(num_samples, 1))  # Current state (Q)
    outputs = simulate_jk_flip_flop(inputs, current_state)  # Simulate JK Flip-Flop behavior

    return inputs, current_state, outputs

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(3,)),  # Input layer (2 for J, K, 1 for Q)
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the neural network
def train_neural_network(inputs, current_state, outputs, epochs=50):
    model = create_model()

    # Combine inputs and current state as the input to the neural network
    input_data = np.column_stack((inputs, current_state))

    # Train the model
    model.fit(input_data, outputs, epochs=epochs)

    return model

# Test the trained model
def test_model(model, test_inputs, test_current_state):
    # Combine test inputs and current state as the input to the neural network
    test_input_data = np.column_stack((test_inputs, test_current_state))

    # Make predictions using the trained model
    predictions = model.predict(test_input_data)

    print("\nTest Results:")
    print("Test Inputs (J, K):", test_inputs)
    print("Test Current State (Q):", test_current_state.flatten())
    print("Predicted Next State (Q+1):", np.round(predictions).flatten())

# Main script
if __name__ == "__main__":
    # Generate training dataset
	print("generating the dataset")
    num_samples = 1000
    train_inputs, train_current_state, train_outputs = generate_dataset(num_samples)

    # Train the neural network
	print("training the model")
    trained_model = train_neural_network(train_inputs, train_current_state, train_outputs)

    # Generate test dataset
	print("testing the model")
    num_test_samples = 10
    test_inputs, test_current_state, _ = generate_dataset(num_test_samples)

    # Test the trained model
    test_model(trained_model, test_inputs, test_current_state)
