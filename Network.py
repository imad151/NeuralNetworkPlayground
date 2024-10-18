import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort

class NeuralNetwork:
    def __init__(self):
        self.UsePretrainedModel = False
        self.weights = []
        self.biases = []
        self.activations = []
        self.learning_rate = 0.01

    def InsertModelArchitecture(self, input_size, activations, hidden_layers, output_size, learning_rate):
        if not self.UsePretrainedModel:
            self.InitParams(input_size, hidden_layers, output_size)
            self.activations = activations
            self.learning_rate = learning_rate

    def InitParams(self, input_size, hidden_layers, output_size):
        previous_layer_size = input_size
        for neurons in hidden_layers:
            weight = np.random.randn(previous_layer_size, neurons) * np.sqrt(2 / (previous_layer_size + neurons))
            bias = np.zeros((1, neurons))
            self.weights.append(weight)
            self.biases.append(bias)
            previous_layer_size = neurons

        weight = np.random.randn(previous_layer_size, output_size) * np.sqrt(2 / (previous_layer_size + output_size))
        bias = np.zeros((1, output_size))
        self.weights.append(weight)
        self.biases.append(bias)

    def ApplyActivation(self, Z, activation):
        if activation == 'Sigmoid':
            return self.Sigmoid(Z)
        elif activation == 'ReLU':
            return self.ReLU(Z)
        elif activation == 'Tanh':
            return self.Tanh(Z)
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def Sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def Tanh(self, Z):
        return np.tanh(Z)

    def SoftMax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def ForwardPass(self, X):
        A = X
        activations = [A]
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.ApplyActivation(Z, self.activations[i])
            activations.append(A)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.SoftMax(Z)
        activations.append(A)
        return A, activations

    def CrossEntropyLoss(self, Y_hat, Y):
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        log_probs = -np.sum(Y * np.log(Y_hat), axis=1)
        return np.mean(log_probs)

    def BackwardsPass(self, activations, X, Y):
        gradients = {
            "dW": [np.zeros_like(w) for w in self.weights],
            "db": [np.zeros_like(b) for b in self.biases]
        }

        A_last = activations[-1]
        dZ = A_last - Y  # SoftMax + Cross-Entropy loss derivative

        for i in reversed(range(len(self.weights))):
            gradients["dW"][i] = np.dot(activations[i].T, dZ) / X.shape[0]
            gradients["db"][i] = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]

            if i > 0:  # Backpropagate through hidden layers
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * (activations[i] > 0).astype(float)  # ReLU derivative

        return gradients

    def UpdateParams(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients['dW'][i]
            self.biases[i] -= self.learning_rate * gradients['db'][i]

    def fit(self, X, Y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            Y_hat, activations = self.ForwardPass(X)
            loss = self.CrossEntropyLoss(Y_hat, Y)
            gradients = self.BackwardsPass(activations, X, Y)
            self.UpdateParams(gradients)

            if verbose and epoch % 10 == 0:
                accuracy = self.GetAccuracy(Y_hat, Y)
                print(f'Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%')

    def GetAccuracy(self, Y_hat, Y):
        predicts = np.argmax(Y_hat, axis=1)
        labels = np.argmax(Y, axis=1)
        return np.mean(predicts == labels) * 100

    def SaveModel(self, file_name):
        self.SaveModelToONNX(file_name)

    def SaveModelToONNX(self, file_name):
      input_info = onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [None, self.weights[0].shape[0]])
      output_info = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [None, self.weights[-1].shape[1]])

      initializer = [
        onnx.helper.make_tensor(f'W_{i}', onnx.TensorProto.FLOAT, w.shape, w.flatten())
        for i, w in enumerate(self.weights)
    ] + [
        onnx.helper.make_tensor(f'B_{i}', onnx.TensorProto.FLOAT, b.shape, b.flatten())
        for i, b in enumerate(self.biases)
    ]

      nodes = []
      input_name = 'input'
      for i in range(len(self.weights)):
        matmul_output = f'layer_{i}_matmul'
        nodes.append(onnx.helper.make_node('MatMul', [input_name, f'W_{i}'], [matmul_output]))

        add_output = f'layer_{i}_output'
        nodes.append(onnx.helper.make_node('Add', [matmul_output, f'B_{i}'], [add_output]))

        if i < len(self.weights) - 1:
            activated_output = f'layer_{i}_activated'
            nodes.append(onnx.helper.make_node('Relu', [add_output], [activated_output]))
            input_name = activated_output
        else:
            input_name = add_output

      nodes.append(onnx.helper.make_node('Softmax', [input_name], ['output'], axis=1))

      graph = onnx.helper.make_graph(nodes, 'NeuralNetwork', [input_info], [output_info], initializer)
    
    # Use opset version 21 explicitly
      model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 21)])
      onnx.save(model, file_name)
      print(f'Model saved to {file_name}')


    def LoadModel(self, file_name):
        session = ort.InferenceSession(file_name)
        print(f'Model loaded from {file_name}')
        return session
