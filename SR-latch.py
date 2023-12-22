import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of an SR latch
def simulate_sr_latch(num_samples):
    states_q = np.zeros((num_samples,), dtype=int)
    states_qbar = np.ones((num_samples,), dtype=int)

    for i in range(1, num_samples):
        # Randomly choose Set (S), Reset (R), or No Change (N) inputs
        input_signal = np.random.choice(['S', 'R', 'N'])

        if input_signal == 'S':
            states_q[i] = 1
            states_qbar[i] = 0
        elif input_signal == 'R':
            states_q[i] = 0
            states_qbar[i] = 1
        else:
            states_q[i] = states_q[i - 1]
            states_qbar[i] = states_qbar[i - 1]

    return states_q, states_qbar

# Function to generate a dataset for the SR latch
def generate_sr_latch_dataset(num_samples):
    return np.zeros((num_samples, 1)), simulate_sr_latch(num_samples)

# Function to create a neural network model for SR latch emulation
def create_sr_latch_model():
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(16, activation='relu', input_shape=(None, 1)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the SR latch
num_samples_sr_latch = 10000
input_data_sr_latch, (outputs_q_sr_latch, outputs_qbar_sr_latch) = generate_sr_latch_dataset(num_samples_sr_latch)

# One-hot encode the output labels
outputs_onehot_sr_latch = np.column_stack((outputs_q_sr_latch, outputs_qbar_sr_latch))

# Reshape the input data for the RNN
input_data_rnn_sr_latch = input_data_sr_latch.reshape((num_samples_sr_latch, 1, 1))

# Split the dataset into training and testing sets
train_input_sr_latch, test_input_sr_latch, train_outputs_sr_latch, test_outputs_sr_latch = train_test_split(
    input_data_rnn_sr_latch, outputs_onehot_sr_latch, test_size=0.2, random_state=42
)

# Create and compile the model
sr_latch_model = create_sr_latch_model()

# Train the model
sr_latch_model.fit(
    train_input_sr_latch,
    train_outputs_sr_latch,
    epochs=50,
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss_sr_latch, test_accuracy_sr_latch = sr_latch_model.evaluate(
    test_input_sr_latch,
    test_outputs_sr_latch,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy_sr_latch * 100:.2f}%')
