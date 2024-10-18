import numpy as np
import pandas as pd
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

# Train the Neural Network using the built-in fit function
nn.fit(
    X=X_train, 
    Y=Y_train_encoded, 
    epochs=100, 
    verbose=True  # Prints progress every 10 epochs as defined in the class
)

nn.SaveModel('MyModel.onnx')
