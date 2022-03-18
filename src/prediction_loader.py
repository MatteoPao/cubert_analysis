import json
import os
import numpy as np
import progressbar as pb

layer_dir = "../results_accuracies/predictions24/"
label_dir = "../results_accuracies/"
layer_name = "Encoder-24-FeedForward-Norm"
RANDOM = True


def main():
    print("Load predictions...")
    prediction = load_prediction()
    label = load_label(label_dir)
    print("Compute and save accuracies...")
    if RANDOM:
        np.random.shuffle(label)
    save_accuracies(prediction, label)


def load_prediction():
    num_files = len(os.listdir(layer_dir))
    out = []
    for index in pb.progressbar(range(num_files)):
        filename = layer_name + "_prediction_" + str(index)+".npy"
        out.append(np.load(layer_dir + filename))

    out = np.asarray(out)
    shape = out.shape
    out.shape = (shape[0]*shape[1], 512, 1024)
    return np.transpose(out)


def load_label(direct):
    return np.load(direct + "label.npy")


def save_accuracies(prediction, label):

    label = np.asarray(label).astype(bool)
    accuracies = []

    for neuron in pb.progressbar(prediction):

        accuracies.append([])
        threshold = np.linspace(neuron.min(), neuron.max(), num=12)[1:11]

        for elem in neuron:
            best = {'acc': 0, 'th': 0}
            for t in threshold:
                elem_b = np.where(elem > t, True, False)
                acc_b = (label & elem_b) | (~label & ~elem_b)
                acc = acc_b.mean()

                if (1 - acc) > acc:
                    acc = 1 - acc

                if acc > best["acc"]:
                    best["acc"] = acc
                    best["th"] = t

            accuracies[-1].append(best)

    # Salvo le accuratezze del layer
    final_out = {'layer': layer_name, 'accuracies': accuracies}
    if RANDOM:
        outF = open("../results_accuracies/E2/" + layer_name + "_random.json", "x")
    else:
        outF = open("../results_accuracies/E2/" + layer_name + "_accuracies.json", "x")
    json_out = json.dumps(final_out, indent=2)
    outF.write(json_out)
    outF.close()


if __name__ == "__main__":
    main()
