from PyQt5.QtCore import QUrl, QProcess
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtWebEngineWidgets import QWebEngineView

import onnxruntime as rt
import numpy as np
import requests
import time
from PIL import Image

class ONNXLoader(QDialog):
    def __init__(self):
        super().__init__()
        print("Initializing ONNXLoader...")
        self.setWindowTitle("Load ONNX Model and Predict")
        self.setGeometry(300, 300, 800, 800)

        # Create UI elements
        self.model_path_input = QLineEdit(self)
        self.image_path_input = QLineEdit(self)
        self.browse_model_button = QPushButton("Browse Model", self)
        self.browse_image_button = QPushButton("Browse Image", self)
        self.load_button = QPushButton("Load Model", self)
        self.predict_button = QPushButton("Predict", self)
        self.image_label = QLabel(self)
        self.result_label = QLabel("Prediction: N/A", self)
        self.web_view = QWebEngineView(self)

        # Layout for model input
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model Path:"))
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(self.browse_model_button)

        # Layout for image input
        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Image Path:"))
        image_layout.addWidget(self.image_path_input)
        image_layout.addWidget(self.browse_image_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(model_layout)
        main_layout.addLayout(image_layout)
        main_layout.addWidget(self.load_button)
        main_layout.addWidget(self.predict_button)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.web_view, stretch=1)  # WebView for Netron

        self.setLayout(main_layout)

        # QProcess for Netron server
        self.server_process = QProcess(self)

        # Connect buttons to actions
        self.browse_model_button.clicked.connect(self.BrowseModel)
        self.browse_image_button.clicked.connect(self.BrowseImage)
        self.load_button.clicked.connect(self.LoadModel)
        self.predict_button.clicked.connect(self.Predict)

    def BrowseModel(self):
        """Open a file dialog to select an ONNX model."""
        print("Opening file dialog for model...")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select ONNX Model", "", "ONNX Files (*.onnx)")
        if file_path:
            print(f"Selected model: {file_path}")
            self.model_path_input.setText(file_path)

    def BrowseImage(self):
        """Open a file dialog to select an image."""
        print("Opening file dialog for image...")
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            print(f"Selected image: {file_path}")
            self.image_path_input.setText(file_path)
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(200, 200))

    def LoadModel(self):
        """Start the Netron server and display the model."""
        self.CloseLoadedModel()
        file_path = self.model_path_input.text()
        if file_path:
            print(f"Loading model: {file_path}")
            self.StartNetron(file_path)

    def Predict(self):
        """Run the selected image through the loaded ONNX model and display the prediction."""
        model_path = self.model_path_input.text()
        image_path = self.image_path_input.text()

        if not model_path or not image_path:
            self.ShowError("Please select both a model and an image.")
            return

        try:
            # Load the ONNX model
            session = rt.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            # Load and preprocess the image
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to 28x28
            image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
            if len(input_shape) == 2:
                image_array = image_array.flatten().reshape(1, 784)

            # Run the model and get the prediction
            prediction = session.run(None, {input_name: image_array})
            predicted_value = np.argmax(prediction[0])
            print(f"Predicted Value: {predicted_value}")

            # Display the prediction
            self.result_label.setText(f"Prediction: {predicted_value}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.ShowError(f"Prediction failed: {str(e)}")

    def StartNetron(self, file_path):
        """Launch the Netron server using QProcess."""
        try:
            netron_executable = r'C:\Users\imado\AppData\Roaming\Python\Python312\Scripts\netron.exe'
            command = f'"{netron_executable}" --host localhost {file_path}'

            self.server_process.setProcessChannelMode(QProcess.MergedChannels)
            self.server_process.readyReadStandardOutput.connect(self.PrintOutput)
            self.server_process.start(command)

            if not self.server_process.waitForStarted():
                print("Netron server failed to start.")
                self.ShowError("Failed to start the Netron server.")
                return

            url = "http://localhost:8080"
            if self.WaitForServer(url):
                print(f"Netron server available at {url}")
                self.web_view.setUrl(QUrl(url))
            else:
                self.ShowError("Failed to connect to the Netron server.")
        except Exception as e:
            self.ShowError(f"An error occurred: {str(e)}")

    def PrintOutput(self):
        """Print output from the Netron server."""
        output = self.server_process.readAllStandardOutput().data().decode()
        print(f"Netron Output: {output}")

    def WaitForServer(self, url, timeout=10):
        """Check if the Netron server is reachable."""
        for i in range(timeout):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        return False

    def ShowError(self, message):
        """Display an error message."""
        QMessageBox.critical(self, "Error", message)

    def CloseLoadedModel(self):
        if hasattr(self, 'session') and self.session is not None:
            print(f'Exiting Netron server')
            self.session = None
        
        if self.server_process.state() == QProcess.Running:
            print(f'Terminating Session')
            self.server_process.terminate()
            self.server_process.waitForFinished()
        else:
            print(f'No Process running')
        self.web_view.reload()


    def closeEvent(self, event):
        """Ensure the Netron server is terminated on close."""
        self.CloseLoadedModel()
        super().closeEvent(event)
