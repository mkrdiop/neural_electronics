import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a 2-to-4 decoder
def simulate_2_to_4_decoder(input_data):
    outputs = np.zeros((len(input_data), 4))

    for i in range(len(outputs)):
        A, B = input_data[i]
        outputs[i] = [int(not A and not B), int(not A and B), int(A and not B), int(A and B)]

    return outputs

# Function to generate a dataset for the 2-to-4 decoder
def generate_2_to_4_decoder_dataset(num_samples):
    input_data = np.random.randint(0, 2, size=(num_samples, 2))  # Random binary inputs (A, B)
    outputs = simulate_2_to_4_decoder(input_data)  # Simulate the 2-to-4 decoder outputs

    return input_data, outputs

# Function to create a neural network model for 2-to-4 decoder emulation
def create_2_to_4_decoder_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # Hidden layer with 8 neurons
        tf.keras.layers.Dense(4, activation='softmax')  # Output layer with 4 neurons for multi-class classification
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the 2-to-4 decoder
num_samples = 10000  # Increase the dataset size
input_data, outputs = generate_2_to_4_decoder_dataset(num_samples)

# One-hot encode the output labels
outputs_onehot = tf.keras.utils.to_categorical(outputs.argmax(axis=1), num_classes=4)

# Split the dataset into training and testing sets
train_input, test_input, train_outputs, test_outputs = train_test_split(
    input_data, outputs_onehot, test_size=0.2, random_state=42
)

# Create and compile the model
decoder_2_to_4_model = create_2_to_4_decoder_model()

# Train the model
decoder_2_to_4_model.fit(
    train_input,
    train_outputs,
    epochs=100,  # Increase the number of epochs
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss, test_accuracy = decoder_2_to_4_model.evaluate(
    test_input,
    test_outputs,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
