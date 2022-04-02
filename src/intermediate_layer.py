# Questo codice è la versione eseguita su macchina
# del notebook colab NeuronPrediction_HiddenLayer
# i commenti è possibile trovarli nel notebook


from keras_bert import load_trained_model_from_checkpoint, get_checkpoint_paths
from cubert.full_cubert_tokenizer import FullCuBertTokenizer, InputExample
from cubert import tokenizer_registry
from typing import Dict, List
import glob
import json
import sys
import numpy as np
import keras.backend as K
import progressbar

def main():
    print(model_path)
    print(data_path)

    paths = get_checkpoint_paths(model_path)

    examples = read_examples_from_json(data_path, "eval")

    tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value, vocab_file=paths.vocab)
    print("Tokenizer loaded")

    #tokenizza il dataset
    features = tokenizer.convert_examples_to_features(examples, [0, 1], 512)
    print("Dataset tokenized")

    # Carica il checkpoint
    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=False)
    # model.summary()

    # Prima alternativa, tronca il modello con keras.backend (Consigliato per risparmiare memoria)
    model = K.function([model.input], [model.get_layer(layer_name).output])
    print("Model loaded")

    inp, seg, mas = [], [], []
    label = []

    for f in features:
        inp.append(f.input_ids)
        seg.append(f.segment_ids)
        mas.append(f.input_mask)
        label.append(f.label_id)

    batched_input = []
    for index in range(int(len(inp) / 8)):
        step = int(index * 8)
        batched_input.append([np.array(inp[step:step + 8]), np.array(mas[step:step + 8])])
    print(np.asarray(batched_input).shape)

    count = 0
    for input_data in progressbar.progressbar(batched_input):
        batched_output = model([input_data, 0])[0]
        with open('results/' + layer_name + '/prediction_' + str(count) + '.npy', 'wb') as file:
            np.save(file, batched_output)
        count += 1


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
    if len(sys.argv) == 4:
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        layer_name = sys.argv[3]
    main()
