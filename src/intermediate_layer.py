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
import keras
import numpy as np

layer_name = "Encoder-10-FeedForward-Norm"

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

    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True,  out_dim=2)
    print("Model loaded")
    #model.summary()

    model = keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    print("Model truncated at layer " + layer_name)
    #model.summary()

    inp = []
    seg = []
    mas = []
    label = []
    for f in features:
        inp.append(f.input_ids)
        seg.append(f.segment_ids)
        mas.append(f.input_mask)
        label.append(f.label_id)

    data_input = [np.array(inp), np.array(seg), np.array(mas)]

    print("\nStarting prediction...")
    pred_results = model.predict(data_input, batch_size=8, verbose=1)
    pred_results = np.transpose(pred_results)

    with open('../results/' + layer_name + '_prediction.npy', 'wb') as file:
        np.save(file, pred_results)



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
