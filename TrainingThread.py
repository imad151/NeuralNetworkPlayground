from PyQt5.QtCore import QThread, pyqtSignal
from Network import NeuralNetwork

class TrainingThread(QThread):
    epoch_update = pyqtSignal(int, float, float)
    TrainingComplete = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.network = None

    def SetupNetwork(self, input_size, hidden_layers, output_size, activations, lr):
        self.network = NeuralNetwork()
        self.network.InsertModelArchitecture(input_size, activations, hidden_layers, output_size, lr)

    def SetupTrainingInstance(self, X, Y, epochs):
        self.X, self.Y, self.epochs = X, Y, epochs

    def run(self):
        for epoch in range(self.epochs):
            Y_hat, activations = self.network.ForwardPass(self.X)
            loss = self.network.CrossEntropyLoss(Y_hat, self.Y)
            accuracy = self.network.GetAccuracy(Y_hat, self.Y)
            self.epoch_update.emit(epoch, loss, accuracy)

            gradients = self.network.BackwardsPass(activations, self.X, self.Y)
            self.network.UpdateParams(gradients)
        self.TrainingComplete.emit()
