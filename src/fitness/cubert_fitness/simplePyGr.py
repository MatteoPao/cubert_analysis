from src.fitness.base_ff_classes.base_ff import base_ff
from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from cubert.full_cubert_tokenizer import FullCuBertTokenizer, CuBertExceptionClassificationProcessor, InputExample
from cubert import tokenizer_registry
import numpy as np

model_path = "fitness/cubert_fitness/cubert_pretrained_model_exceptions"


class simplePyGr(base_ff):
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
    maximise = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.u_id = 0

        paths = get_checkpoint_paths(model_path)
        self.model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128)
        print("model loaded")

        self.tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value, vocab_file=paths.vocab)
        print("tokenizer loaded")

        processor = CuBertExceptionClassificationProcessor()
        self.label_list = processor.get_labels()

    def evaluate(self, ind, **kwargs):

        guid = "eval" + str(self.u_id)
        self.u_id += 1

        guess_nt = InputExample(guid=guid, text_a=ind.phenotype, text_b=None, label="None")
        guess = self.tokenizer.convert_single_example(guess_nt, self.label_list, 128)

        prediction = self.model.predict([np.expand_dims(np.array(guess.input_ids), axis=0),
                                         np.expand_dims(np.array(guess.segment_ids), axis=0),
                                         np.expand_dims(np.array(guess.input_mask), axis=0)])

        return prediction[0][0] * 100