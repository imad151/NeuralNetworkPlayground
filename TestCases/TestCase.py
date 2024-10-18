import numpy as np
from Network import NeuralNetwork
import onnxruntime

def test_neural_network():
    nn = NeuralNetwork()
    input_size = 3
    hidden_layers = [5, 4]
    output_size = 2
    activations = ['ReLU', 'ReLU', 'SoftMax']
    learning_rate = 0.01

    nn.InsertModelArchitecture(
        input_size=input_size,
        activations=activations,
        hidden_layers=hidden_layers,
        output_size=output_size,
        learning_rate=learning_rate
    )

    assert len(nn.weights) == 3, "Incorrect number of weight layers."
    assert len(nn.biases) == 3, "Incorrect number of bias layers."

    X = np.array([[0.1, 0.2, 0.3]])
    Y_hat = nn.ForwardPass(X)

    assert Y_hat.shape == (1, output_size), "Incorrect output shape from forward pass."

    Z = np.array([[1, -1], [2, -2]])
    assert np.all(nn.ReLU(Z) == np.array([[1, 0], [2, 0]])), "ReLU activation failed."
    assert np.allclose(nn.Sigmoid(Z), 1 / (1 + np.exp(-Z))), "Sigmoid activation failed."

    Y_true = np.array([[1, 0]])
    loss = nn.CrossEntropyLoss(Y_hat, Y_true)
    assert loss > 0, "Loss should be greater than 0."

    gradients = nn.BackwardsPass(X, Y_true)
    assert len(gradients['dW']) == 3, "Incorrect number of weight gradients."
    assert len(gradients['db']) == 3, "Incorrect number of bias gradients."

    old_weights = [w.copy() for w in nn.weights]
    nn.UpdateParams(gradients)
    assert not np.array_equal(old_weights[0], nn.weights[0]), "Weights not updated properly."

    X_train = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    Y_train = np.array([[1, 0], [0, 1]])
    nn.fit(X_train, Y_train, epochs=10, verbose=False)

    Y_hat = nn.ForwardPass(X_train)
    accuracy = nn.GetAccuracy(Y_hat, Y_train)
    print(accuracy)
    assert accuracy >= 50.0, "Accuracy too low, model might not be working correctly."

    print("All tests passed.")


def test_xor():
    nn = NeuralNetwork()
    
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input: XOR truth table
    Y_train = np.array([[0], [1], [1], [0]])  # Output: XOR results

    # One-hot encode the target values for use with SoftMax activation
    Y_train_encoded = np.zeros((Y_train.size, 2))
    Y_train_encoded[np.arange(Y_train.size), Y_train.flatten()] = 1

    nn.InsertModelArchitecture(
        input_size=2,          # Two input features (XOR inputs)
        activations=['ReLU', 'SoftMax'],  # Hidden layer with ReLU, output layer with SoftMax
        hidden_layers=[2],     # One hidden layer with 2 neurons
        output_size=2,         # Two outputs (for binary classification)
        learning_rate=0.1      # A slightly higher learning rate for quick training
    )

    nn.fit(X_train, Y_train_encoded, epochs=1000, verbose=False)

    Y_hat = nn.ForwardPass(X_train)  # Get predictions
    predictions = np.argmax(Y_hat, axis=1).reshape(-1, 1)  # Convert to class labels

    # Check if predictions match the expected XOR output
    assert np.array_equal(predictions, Y_train), f"XOR test failed! Predictions: {predictions}"

    print("XOR test passed! Neural network successfully learned XOR function.")

def train_and_save_model():
    # Step 1: Initialize and Train the Neural Network
    nn = NeuralNetwork()

    # XOR Data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([[0], [1], [1], [0]])

    # One-hot encode the target values
    Y_train_encoded = np.zeros((Y_train.size, 2))
    Y_train_encoded[np.arange(Y_train.size), Y_train.flatten()] = 1

    # Define and train the network
    nn.InsertModelArchitecture(
        input_size=2,          # Two input features
        activations=['ReLU', 'SoftMax'],  # Hidden layer with ReLU, output with SoftMax
        hidden_layers=[2],     # One hidden layer with 2 neurons
        output_size=2,         # Two output classes (binary classification)
        learning_rate=0.1      # Learning rate
    )

    # Train the network
    nn.fit(X_train, Y_train_encoded, epochs=1000, verbose=False)

    # Save the trained model
    model_file = "xor_model.onnx"
    nn.SaveModel(model_file)

def load_and_use_model():
    # Load the model
    model_file = "xor_model.onnx"
    session = onnxruntime.InferenceSession(model_file)

    # XOR Test Data
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Make predictions one input at a time
    predictions = []
    for i in range(X_test.shape[0]):
        X_single = X_test[i].reshape(1, -1).astype(np.float32)  # Reshape to (1, 2)

        # Prepare input dictionary
        inputs = {session.get_inputs()[0].name: X_single}
        
        # Run inference and collect the output
        outputs = session.run(None, inputs)
        prediction = np.argmax(outputs[0], axis=1).item()
        predictions.append(prediction)

    # Print all predictions
    print(f"Predictions: {predictions}")

