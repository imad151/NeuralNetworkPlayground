import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Network import NeuralNetwork

# Load and preprocess the MNIST dataset
def load_mnist_from_csv(file_path):
    """Load and preprocess the MNIST dataset from a CSV file."""
    data = pd.read_csv(file_path, header=None)  # Load CSV without headers

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

    # Extract labels and features
    labels = data.iloc[:, 0].values  # First column: labels
    features = data.iloc[:, 1:].values  # Remaining columns: pixel values

    # Normalize pixel values to the range [0, 1]
    features = features / 255.0

    # One-hot encode the labels
    one_hot_labels = np.zeros((labels.size, 10))  # 10 classes (digits 0-9)
    one_hot_labels[np.arange(labels.size), labels] = 1

    return features, one_hot_labels

# Load and shuffle the MNIST dataset
X_train, Y_train_encoded = load_mnist_from_csv('datasets/mnist_train.csv')

# Initialize the Neural Network
nn = NeuralNetwork()
nn.InsertModelArchitecture(
    input_size=784,          # 28x28 pixels
    activations=['ReLU', 'ReLU', 'SoftMax'],  # Hidden layers: ReLU, Output: SoftMax
    hidden_layers=[16, 16],  # Two hidden layers with 16 neurons each
    output_size=10,          # 10 classes (digits 0-9)
    learning_rate=0.01        # Learning rate
)

# Set up real-time plot
plt.ion()  # Enable interactive mode
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)', color='tab:orange')

loss_line, = ax1.plot([], [], 'b-', label='Loss')
acc_line, = ax2.plot([], [], 'r-', label='Accuracy')

# Initialize lists to track metrics
epochs = []
losses = []
accuracies = []

def update_plot(epoch, loss, accuracy):
    """Update the plot with new epoch data."""
    epochs.append(epoch)
    losses.append(loss)
    accuracies.append(accuracy)

    loss_line.set_data(epochs, losses)
    acc_line.set_data(epochs, accuracies)

    ax1.relim()
    ax2.relim()
    ax1.autoscale_view()
    ax2.autoscale_view()

    plt.pause(0.1)  # Small pause to allow the plot to update

# Training loop with real-time plotting
num_epochs = 1000
for epoch in range(num_epochs):
    # Shuffle data at the beginning of each epoch
    shuffled_indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[shuffled_indices]
    Y_train_shuffled = Y_train_encoded[shuffled_indices]

    # Forward pass
    Y_hat = nn.ForwardPass(X_train_shuffled)
    loss = nn.CrossEntropyLoss(Y_hat, Y_train_shuffled)
    accuracy = nn.GetAccuracy(Y_hat, Y_train_shuffled)

    # Backpropagation and parameter update
    gradients = nn.BackwardsPass(X_train_shuffled, Y_train_shuffled)
    nn.UpdateParams(gradients)

    # Print progress
    if epoch % 10 == 0:
        print(f'Epoch = {epoch + 1}, loss = {loss:.4f}, accuracy = {accuracy:.2f}%')

    # Update the plot in real-time
        update_plot(epoch + 1, loss, accuracy)

# Finalize the plot
plt.ioff()  # Disable interactive mode
plt.show()
