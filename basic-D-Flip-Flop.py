import tensorflow as tf
import numpy as np

# Function to simulate D Flip-Flop behavior
def simulate_d_flip_flop(inputs, current_state):
    return inputs

# Generate training dataset for D Flip-Flop
def generate_dataset(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 1))  # Input data (0 or 1)
    current_state = np.random.randint(0, 2, size=(num_samples, 1))  # Current state (0 or 1)
    outputs = simulate_d_flip_flop(inputs, current_state)  # Simulate D Flip-Flop behavior

    return inputs, current_state, outputs

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='relu', input_shape=(2,)),  # Input layer
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
    print("Test Inputs:", test_inputs.flatten())
    print("Test Current State:", test_current_state.flatten())
    print("Predicted Outputs:", predictions.flatten())

# Main script
if __name__ == "__main__":
    # Generate training dataset
    num_samples = 1000
    train_inputs, train_current_state, train_outputs = generate_dataset(num_samples)

    # Train the neural network
    trained_model = train_neural_network(train_inputs, train_current_state, train_outputs)

    # Generate test dataset
    num_test_samples = 10
    test_inputs, test_current_state, _ = generate_dataset(num_test_samples)

    # Test the trained model
    test_model(trained_model, test_inputs, test_current_state)
