from PyQt5 import uic
from PyQt5.QtWidgets import *


class LayerConfig(QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("UI/LayerConfig.ui", self)

        self.ActivationFunc = self.findChild(QComboBox, "comboBox")
        self.LayerSize = self.findChild(QSpinBox, "LayerSizeSpinbox")
        