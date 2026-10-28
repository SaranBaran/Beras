import numpy as np

from beras.core import Diffable, Tensor


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
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        outputs = self.inputs[0].size
        dinputs = -2 * (y_true - y_pred) / outputs
        return [Tensor(dinputs), Tensor(np.zeros_like(y_true))]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        n = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / n
        return Tensor(loss)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        eps = 1e-7
        y_pred = np.clip(self.inputs[0], eps, 1 - eps)
        y_true = self.inputs[1]
        n = y_true.shape[0]
        dinputs = -y_true / (n * y_pred)
        return [Tensor(dinputs), Tensor(np.zeros_like(y_true))]
