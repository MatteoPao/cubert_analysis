from fitness.base_ff_classes.base_ff import base_ff
from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from cubert.full_cubert_tokenizer import FullCuBertTokenizer, CuBertExceptionClassificationProcessor, InputExample
from cubert import tokenizer_registry
import numpy as np

model_path = "fitness/cubert_fitness/cubert_pretrained_model_exceptions"


class exceptionClassification(base_ff):

    maximise = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.u_id = 0

        paths = get_checkpoint_paths(model_path)
        self.model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128, out_dim=21)
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

        return prediction[0][1] * 100