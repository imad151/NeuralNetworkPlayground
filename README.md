# **NeuralNetwork Playground**

An easy-to-use **GUI-based project** for creating simple, deployable neural networks on the go. This package takes inspiration from the [TensorFlow Playground](https://playground.tensorflow.org/), providing a local environment to build, train, and examine models.

---

## **Features**
- **Set up Model Architecture**: Easily configure neural networks using a simple interface.
- **Quick Training**: Train networks rapidly with an interactive GUI.
- **Monitor Training Process**: Visualize real-time training progress and model behavior.
- **ONNX Model Loader**: Load and inspect pre-trained ONNX models for deeper insights.

---

## **Table of Contents**
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)

---

## **Installation**
### **Clone the Repository and Install Dependencies**
```bash
git clone https://github.com/imad151/NeuralNetworkPlayground.git
cd NeuralNetworkPlayground
pip install -r requirements.txt
```

### **Install Netron for Model Visualization**
You need **Netron** to visualize your models. Install it via the command line:
```bash
pip install netron
```

### **Append the Netron Path to the Code**
Update the path to the Netron executable in `LoadNetwork.py` (around line 127) like this:
```python
netron_executable = r'C:\Users\[user]\AppData\Roaming\Python\Python312\Scripts\netron.exe'
```
> Replace `[user]` with your actual username or adjust the path if it differs on your system.

---

## **Usage**
1. **Run the GUI Application**  
   Launch the main GUI by running:
   ```bash
   python main.py
   ```

2. **Select a Dataset**  
   Choose a dataset to begin building your neural network.

3. **Setup a Model Architecture**  
   Configure your neural network layers and settings through the interface.

4. **Train the Model**  
   Observe the training progress and save the trained model.

5. **Save and Load Models**  
   - Trained models are automatically saved to the `ProjectFolder\Models` directory.
   - Use the **"Load Model"** button to load an ONNX model for inspection or further training.


---

## **Datasets**

- Currently, No dataset is directly shipped so you'll have to download one on your own.
- Download MNIST dataseta [here](https://drive.google.com/file/d/1eEKzfmEu6WKdRlohBQiqi3PhW_uIVJVP/view?usp=sharing/)
- Rename it to `mnist_train.csv` and save it to `ProjectFolder/Datasets/`.
---

## **Contact**
For questions or collaboration inquiries, feel free to reach out:  
- **GitHub**: [imad151](https://github.com/imad151)

---

## **Future Improvements (Optional Section)**
- Support for more datasets.
- Support for CNNs, GANs and other model types
- Dataset and Layer Normalization
- Implement new visualizations for training metrics.
- Add exporting to formats other than ONNX.
