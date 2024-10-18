import sys
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

from UI.LayerConfig import LayerConfig

from Network import NeuralNetwork
from TrainingThread import TrainingThread
from LoadModel import ONNXLoader

from sklearn.preprocessing import OneHotEncoder  # Only for some pre-processing

class ControlPanelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI/ControlPanel.ui", self)
        self.MaxLayers = 5
        self.NumLayers = 0
        self.LayerWidgets = []

        # Initialize the ONNX loader and training thread
        self.onnx_loader = ONNXLoader()
        self.NetworkThread = TrainingThread()

        self.fig, self.ax = None, None
        self.loss_data, self.accuracy_data, self.epoch_data = [], [], []
        curr_date_time = datetime.now().strftime(r"%Y%m%d_%H%M")
        self.model_name = f"trained_model {curr_date_time}.onnx"

        # UI initialization
        self.InitUI()
        self.ConnectSignals()

    def InitUI(self):
        self.AddLayerButton = self.findChild(QPushButton, "AddButton")
        self.SubmitButton = self.findChild(QPushButton, "TrainButton")
        self.RemoveButton = self.findChild(QPushButton, "RemoveButton")
        self.LoadButton = self.findChild(QPushButton, "LoadButton")

        self.InputSizeSpinbox = self.findChild(QSpinBox, "InputSizeSpinbox")
        self.OutputSizeSpinbox = self.findChild(QSpinBox, "OutputSizeSpinbox")
        self.EpochSpinbox = self.findChild(QSpinBox, "EpochSpinbox")
        self.DatasetCombobox = self.findChild(QComboBox, "DatasetBox")
        self.LearningRateSpinbox = self.findChild(QDoubleSpinBox, "LearningRateSpinbox")

        self.LayerContainer = self.findChild(QWidget, "LayerContainer")
        self.ContainerLayout = QVBoxLayout(self.LayerContainer)

    def ConnectSignals(self):
        self.AddLayerButton.pressed.connect(self.AddAdditionalLayerConfig)
        self.SubmitButton.pressed.connect(self.SubmitConfig)
        self.RemoveButton.pressed.connect(self.RemoveLastLayer)
        self.LoadButton.pressed.connect(self.OpenONNXLoader)

        self.NetworkThread.epoch_update.connect(self.UpdatePlot)
        self.NetworkThread.TrainingComplete.connect(self.OnTrainingComplete)

    def OpenONNXLoader(self):
        self.onnx_loader.show()

    def InitializePlot(self):
        if not self.fig:
            self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
            plt.ion()
        self.loss_data, self.accuracy_data, self.epoch_data = [], [], []
        plt.show()

    def UpdatePlot(self, epoch, loss, accuracy):
        self.epoch_data.append(epoch)
        self.loss_data.append(loss)
        self.accuracy_data.append(accuracy)

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].plot(self.epoch_data, self.loss_data, label='Loss')
        self.ax[0].set_title('Loss over Epochs')
        self.ax[1].plot(self.epoch_data, self.accuracy_data, label='Accuracy', color='green')
        self.ax[1].set_title('Accuracy over Epochs')
        plt.pause(0.01)

    def OnTrainingComplete(self):
        print("Training Completed!")
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        self.SaveModel()

    def SaveModel(self):
        os.makedirs("Models", exist_ok=True)
        model_path = os.path.join("Models", self.model_name)
        self.NetworkThread.network.SaveModel(model_path)
        print(f"Model saved to {model_path}")

    def AddAdditionalLayerConfig(self):
        if self.NumLayers < self.MaxLayers:
            new_layer = LayerConfig()
            self.ContainerLayout.addWidget(new_layer)
            self.LayerWidgets.append(new_layer)
            self.NumLayers += 1
        else:
            print("Max Layers Reached")

    def RemoveLastLayer(self):
        if self.LayerWidgets:
            widget = self.LayerWidgets.pop()
            self.ContainerLayout.removeWidget(widget)
            widget.deleteLater()
            self.NumLayers -= 1

    def SubmitConfig(self):
        self.InitializePlot()
        config = [{"activation": layer.ActivationFunc.currentText(), "layer_size": layer.LayerSize.value()}
                  for layer in self.LayerWidgets]

        hidden_layers = [c["layer_size"] for c in config]
        activations = [c["activation"] for c in config] + ["SoftMax"]

        self.SetupNetwork(hidden_layers, activations)

    def SetupNetwork(self, hidden_layers, activations):
        input_size, output_size = self.InputSizeSpinbox.value(), self.OutputSizeSpinbox.value()
        lr, epochs = self.LearningRateSpinbox.value(), self.EpochSpinbox.value()
        dataset = pd.read_csv('Datasets/mnist_train.csv').values.T

        X, Y = dataset[1:].T / 255.0, OneHotEncoder(sparse_output=False).fit_transform(dataset[0].reshape(-1, 1))
        self.NetworkThread.SetupNetwork(input_size, hidden_layers, output_size, activations, lr)
        self.NetworkThread.SetupTrainingInstance(X, Y, epochs)
        self.NetworkThread.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ControlPanelGUI()
    window.show()
    sys.exit(app.exec_())
