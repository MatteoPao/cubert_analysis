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

"""
paths = get_checkpoint_paths(model_path)
print(paths)
model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128, out_dim=2)
print("model loaded")

tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value, vocab_file=paths.vocab)
print("tokenizer loaded")

processor = CuBertVariableMisuseProcessor()
label_list = processor.get_labels()

guess_nt = InputExample(guid="test-0", text_a=ind.phenotype, text_b=None, label="Correct")
guess = tokenizer.convert_single_example(guess_nt, label_list, 128)

prediction = model.predict([np.expand_dims(np.array(guess.input_ids), axis=0),
                            np.expand_dims(np.array(guess.segment_ids), axis=0),
                            np.expand_dims(np.array(guess.input_mask), axis=0)])

print(prediction[0][0] * 100)


#-----------------------------------------------------------------------------------


if len(sys.argv) == 3:
    model_path = sys.argv[1]
    data_path = sys.argv[2]

examples = read_examples_from_jsonls(data_path, "eval")
s_f = list()
s_f.append(tokenizer.convert_single_example(examples[0], label_list, 128))
s_f.append(tokenizer.convert_single_example(examples[1], label_list, 128))
s_f.append(tokenizer.convert_single_example(examples[2], label_list, 128))
print(np.array(s_f).shape)

print("STAMPA ARRAY COMPLETO")
print([np.expand_dims(np.array(s_f[0].input_ids), axis=0), np.expand_dims(np.array(s_f[0].segment_ids), axis=0), np.expand_dims(np.array(s_f[0].input_mask), axis=0)])
print([np.expand_dims(np.array(s_f[0].input_ids), axis=0).shape, np.expand_dims(np.array(s_f[0].segment_ids), axis=0).shape, np.expand_dims(np.array(s_f[0].input_mask), axis=0).shape])

predicts = list()
predicts.append(model.predict([np.expand_dims(np.array(s_f[0].input_ids), axis=0),
                               np.expand_dims(np.array(s_f[0].segment_ids), axis=0),
                               np.expand_dims(np.array(s_f[0].input_mask), axis=0)]))
predicts.append(model.predict([np.expand_dims(np.array(s_f[1].input_ids), axis=0),
                               np.expand_dims(np.array(s_f[1].segment_ids), axis=0),
                               np.expand_dims(np.array(s_f[1].input_mask), axis=0)]))
predicts.append(model.predict([np.expand_dims(np.array(s_f[2].input_ids), axis=0),
                               np.expand_dims(np.array(s_f[2].segment_ids), axis=0),
                               np.expand_dims(np.array(s_f[2].input_mask), axis=0)]))

print(predicts[0])
print(predicts[1])
print(predicts[2])
"""


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

    feature = tokenizer.convert_single_example(examples[0], [0, 1], 128)

    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128, out_dim=2)
    print("Model loaded")

    layer_name = "Encoder-23-FeedForward-Norm"

    int_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.get_layer(layer_name).output)
    int_layer_model.summary()
    int_output = int_layer_model.predict([np.expand_dims(np.array(feature.input_ids), axis=0),
                                          np.expand_dims(np.array(feature.segment_ids), axis=0),
                                          np.expand_dims(np.array(feature.input_mask), axis=0)])

    print(int_output)

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
