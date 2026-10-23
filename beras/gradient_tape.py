from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}
        while queue:
            curr_node = queue.pop()
            layers = self.previous_layers[id(curr_node)]
            input_gradients = self.compose_input_gradients(grads[id(curr_node)])
            weight_gradients = self.compose_weight_gradients(grads[id(curr_node)])
            for input, input_grad in zip(layers.inputs, input_gradients):
                queue.append(input)
                grads[input] = input_grad
            for weight, weight_grad in zip(layers.weights, weight_gradients):
                grads[weight] = weight_grad
        
        grad_of_sources = []
        for source in sources:
            grad_of_sources.append(Tensor(grads[source]))

        return grad_of_sources

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
