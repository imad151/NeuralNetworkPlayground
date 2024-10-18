from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from PyQt5 import uic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np



class PlotHandler(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('UI/PerformancePlot.ui', self)

        self.InitUI()
        self.InitPlot()

    def InitUI(self):
        self.PlotWidget = self.findChild(QGraphicsView, "PerformancePlot")
        self.Scene = QGraphicsScene()
        self.PlotWidget.setScene(self.Scene)

    def InitPlot(self):
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Accuracy', color='b')
        self.ax2.set_ylabel('Loss', color='r')
        self.ax1.tick_params(axis='y', labelcolor='b')
        self.ax2.tick_params(axis='y', labelcolor='r')

        self.line_acc, = self.ax1.plot([], [], 'b-o', label='Accuracy')
        self.line_loss, = self.ax2.plot([], [], 'r-o', label='Loss')
        
        self.canvas = FigureCanvas(self.fig)

    def update_plot(self, accuracies, losses):
        print('Updating plot')
        epochs = len(accuracies) * 10

        self.line_acc.set_xdata(np.arange(0, epochs * 10, 10))
        self.line_acc.set_ydata(accuracies)

        self.line_loss.set_xdata(np.arange(0, epochs * 10, 10))
        self.line_loss.set_ydata(losses)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        self.canvas.draw()

        # Convert the canvas to a QPixmap to be displayed in the QGraphicsView
        width, height = self.fig.canvas.get_width_height()
        qimg = QImage(self.canvas.buffer_rgba(), int(width), int(height), QImage.Format_RGBA8888)
        img = QPixmap.fromImage(qimg)

        # Update the QGraphicsScene with the new image
        self.Scene.clear()
        self.Scene.addPixmap(img)
        self.PlotWidget.fitInView(self.Scene.sceneRect(), mode=1)
