import onnx
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Load the ONNX model
onnx_model_path = r"D:\Python\ML\NeuralNetwork\NN\Models\feastconv_Opset18.onnx"  # Replace with your model path
model = onnx.load(onnx_model_path)

# Initialize ONNX Runtime session
session = ort.InferenceSession(onnx_model_path)

# Inspect model inputs
input_names_and_types = [(inp.name, inp.shape, inp.type) for inp in session.get_inputs()]
print(f"Model Inputs: {input_names_and_types}")

# Prepare input data with appropriate data types
input_feed = {}
for name, shape, dtype in input_names_and_types:
    # Replace 'None' with 1 for dynamic dimensions, if any
    shape = [1 if dim is None else dim for dim in shape]

    # Generate input data based on expected type
    if dtype == 'tensor(int64)':
        input_feed[name] = np.random.randint(0, 10, size=shape).astype(np.int64)
    elif dtype == 'tensor(float)':
        input_feed[name] = np.random.randn(*shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input type: {dtype}")

# Function to visualize activations
def visualize_activations(activations, layer_name):
    plt.figure(figsize=(10, 5))
    plt.imshow(activations, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(f"Activations for Layer: {layer_name}")
    plt.show()

# Get the names of all layers (including intermediate and output layers)
layer_names = [output.name for output in session.get_outputs()]

# Run inference with all required inputs
activations = session.run(layer_names, input_feed)

# Visualize activations for each layer
for i, activation in enumerate(activations):
    layer_name = f"Layer {i + 1} - {layer_names[i]}"
    # Flatten activation for easier visualization
    activation_flat = activation.reshape(-1, activation.shape[-1])
    visualize_activations(activation_flat, layer_name)
