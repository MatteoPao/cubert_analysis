import numpy as np
from tensorflow import keras
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff

FILE_NAME = "../model/04102021-174838_model"

class testNN(base_ff):
    """Fitness function"""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self._model = keras.models.load_model("./model/" + FILE_NAME)
        self._input_size = self._model.layers[0].input_shape[0][1]
        
        # Set target string.
        #self.target = params['TARGET']

    def evaluate(self, ind, **kwargs):
        guess = ind.phenotype
        distance = abs(len(guess) - self._input_size)

        fitness = distance

        # Prepare the input
        inputFF = np.float_(list(guess))
        inputFF = np.resize(inputFF, (1, self._input_size))
        inputFF[0][len(guess):] = 0

        pred = self._model.predict(inputFF)
        res = np.argmax(pred[0])

        fitness += res

        return fitness
