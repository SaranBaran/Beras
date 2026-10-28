import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.dictionary = {}
        labels = np.unique(data)
        labels_matrix = np.eye(labels.size)
        for i in range(labels.size):
            self.dictionary[labels[i]] = labels_matrix[i]

    def forward(self, data):
        if getattr(self, "dictionary", None) is None:
            self.fit(data)
        labels_array = []
        for d in data:
            labels_array.append(self.dictionary[d])
        return np.array(labels_array)

    def inverse(self, data):
        labels_array = []
        for d in data:
            keys = [key for key, val in self.dictionary.items() if np.array_equal(val, d)]
            labels_array.append(keys[0])
        return np.array(labels_array)
