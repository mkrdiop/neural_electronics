import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to simulate the behavior of a D latch
def simulate_d_latch(num_samples):
    states_q = np.zeros((num_samples,), dtype=int)
    states_qbar = np.ones((num_samples,), dtype=int)

    for i in range(1, num_samples):
        # Randomly choose Data (D) input
        input_signal_d = np.random.choice([0, 1])

        # Randomly choose rising or falling clock edge
        clock_edge = np.random.choice(['Rising', 'Falling'])

        # Update state based on clock edge
        if clock_edge == 'Rising':
            states_q[i] = input_signal_d
            states_qbar[i] = 1 - input_signal_d
        elif clock_edge == 'Falling':
            states_q[i] = states_q[i - 1]
            states_qbar[i] = states_qbar[i - 1]

    return states_q, states_qbar

# Function to generate a dataset for the D latch
def generate_d_latch_dataset(num_samples):
    return np.zeros((num_samples, 1)), simulate_d_latch(num_samples)

# Function to create a neural network model for D latch emulation
def create_d_latch_model():
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(16, activation='relu', input_shape=(None, 1)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate a dataset for the D latch
num_samples_d_latch = 10000
input_data_d_latch, (outputs_q_d_latch, outputs_qbar_d_latch) = generate_d_latch_dataset(num_samples_d_latch)

# One-hot encode the output labels
outputs_onehot_d_latch = np.column_stack((outputs_q_d_latch, outputs_qbar_d_latch))

# Reshape the input data for the RNN
input_data_rnn_d_latch = input_data_d_latch.reshape((num_samples_d_latch, 1, 1))

# Split the dataset into training and testing sets
train_input_d_latch, test_input_d_latch, train_outputs_d_latch, test_outputs_d_latch = train_test_split(
    input_data_rnn_d_latch, outputs_onehot_d_latch, test_size=0.2, random_state=42
)

# Create and compile the model
d_latch_model = create_d_latch_model()

# Train the model
d_latch_model.fit(
    train_input_d_latch,
    train_outputs_d_latch,
    epochs=50,
    verbose=1,
    validation_split=0.2
)

# Evaluate the model on the test set
test_loss_d_latch, test_accuracy_d_latch = d_latch_model.evaluate(
    test_input_d_latch,
    test_outputs_d_latch,
    verbose=0
)

print(f'Test Accuracy: {test_accuracy_d_latch * 100:.2f}%')
