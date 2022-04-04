from fitness.base_ff_classes.base_ff import base_ff
from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from cubert.full_cubert_tokenizer import FullCuBertTokenizer, CuBertVariableMisuseProcessor, InputExample
from cubert import tokenizer_registry

import keras.backend as K
import numpy as np

model_path = "fitness/cubert_fitness/cubert_pretrained_model_epochs_2"
layer_name = "Encoder-5-FeedForward-Norm"
neuron = 803  # Indice neurone
elem = 0  # Indice individuo
th = -0.179
inv = True


class intermediate_layer_e2(base_ff):
    """
    Basic fitness function template for writing new fitness functions. This
    basic template inherits from the base fitness function class, which
    contains various checks and balances.

    Note that all fitness functions must be implemented as a class.

    Note that the class name must be the same as the file name.

    Important points to note about base fitness function class from which
    this template inherits:

      - Default Fitness values (can be referenced as "self.default_fitness")
        are set to NaN in the base class. While this can be over-written,
        PonyGE2 works best when it can filter solutions by NaN values.

      - The standard fitness objective of the base fitness function class is
        to minimise fitness. If the objective is to maximise fitness,
        this can be over-written by setting the flag "maximise = True".

    """

    # The base fitness function class is set up to minimise fitness.
    # However, if you wish to maximise fitness values, you only need to
    # change the "maximise" attribute here to True rather than False.
    # Note that if fitness is being minimised, it is not necessary to
    # re-define/overwrite the maximise attribute here, as it already exists
    # in the base fitness function class.
    maximise = not inv

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        paths = get_checkpoint_paths(model_path)

        self.worst_ind = 0

        out_f = open("over_th.txt", "x")
        out_f.write("Individui migliori del threshold -------\n")
        out_f.close()

        # Carica il modello
        self.model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=False)
        self.model = K.function([self.model.input], [self.model.get_layer(layer_name).output])
        print("model loaded")

        self.tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value,
                                             vocab_file=paths.vocab)
        print("tokenizer loaded")

        print("------ Impostazioni ------")
        print("Neurone: ", neuron)
        print("Elem: ", elem)
        print("maximize: ", self.maximise)

    def evaluate(self, ind, **kwargs):

        token = self.tokenizer.tokenize(ind.phenotype)
        if len(token) > 512:
            return -2

        input_ids = self.tokenizer.convert_tokens_to_ids(token)
        while len(input_ids) < 512:
            input_ids.append(0)

        ind_input = [np.array(input_ids, ndmin=2), np.zeros((1, 512))]

        ind_output = self.model([ind_input, 0])[0][0]

        # return ind_output[elem][neuron]

        res = ind_output[elem][neuron]
        if (self.maximise and res < self.worst_ind) or (not self.maximise and res > self.worst_ind):
            self.worst_ind = res

        if (self.maximise and res > th) or (not self.maximise and res < th):
            out_f = open("over_th.txt", "a")
            out_f.write(str(res) + "\n")
            out_f.write(ind.phenotype)
            out_f.write("\n -----------------------------\n")
            out_f.close()

        return res
