import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.square(np.subtract(y_true, y_pred)).mean())

    def get_input_gradients(self) -> list[Tensor]:
        samples = len(self.inputs)
        outputs = len(self.inputs[0])
        self.dinputs = -2 * (y_true - self.inputs) / outputs
        self.dinputs = self.dinputs / samples
        return NotImplementedError

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
