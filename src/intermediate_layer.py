from fitness.base_ff_classes.base_ff import base_ff
from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from cubert.full_cubert_tokenizer import FullCuBertTokenizer, CuBertVariableMisuseProcessor, InputExample
from cubert import tokenizer_registry
from typing import Dict, List
import glob
import json
import sys
import keras
import numpy as np

def main():
    print(model_path)
    print(data_path)

    paths = get_checkpoint_paths(model_path)

    examples = read_examples_from_json(data_path, "eval")

    tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value, vocab_file=paths.vocab)
    print("Tokenizer loaded")

    #tokenizza tutto il dataset
    #features = tokenizer.convert_examples_to_features(examples, [0, 1], 128)
    #print(features[0].input_ids)

    feature = tokenizer.convert_single_example(examples[5], [0, 1], 128)

    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128, out_dim=2)
    print("Model loaded")
    #model.summary()

    layer_name = "Encoder-23-FeedForward-Norm"
    model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    #model.summary()

    int_output = model.predict([np.expand_dims(np.array(feature.input_ids), axis=0),
                                np.expand_dims(np.array(feature.segment_ids), axis=0),
                                np.expand_dims(np.array(feature.input_mask), axis=0)])

    for index, line in enumerate(int_output[0]):
        print(index, line)
        print("Minimo: ", min(line))
        print("Massimo: ", max(line))

    print(np.array(int_output[0]).shape)
    print("MinimoAssoluto: ", np.array(int_output[0]).min())
    print("MassimoAssoluto: ", np.array(int_output[0]).max())



def read_json(input_file):
    f = open(input_file, 'r')
    lines = f.readlines()
    for line in lines:
        yield json.loads(line)


def create_examples(raw_examples, set_type):
    examples = []
    for (i, example) in enumerate(raw_examples):
        guid = "%s-%s" % (set_type, i)

        text_a = example["function"]
        label = example["label"]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def read_examples_from_json(data_dir, set_type):
    examples: List[Dict[str, int]] = []
    for file in glob.glob(data_dir):
        examples.extend(read_json(file))
    return create_examples(examples, set_type)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_path = sys.argv[1]
        data_path = sys.argv[2]
    main()
