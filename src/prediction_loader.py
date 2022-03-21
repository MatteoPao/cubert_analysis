'''
Questo script permette di caricare e unificare i risultati
generati con colab in un unico numpy array.
Successivamente è possibile calcolare le accuratezze dei
layer e salvare il tutto in un file json
'''
import json
import os
import numpy as np
import progressbar as pb

# Modificare in base alle proprie esigenze
layer_dir = "../results_accuracies/predictions24/"
label_dir = "../results_accuracies/"
layer_name = "Encoder-24-FeedForward-Norm"

# Se True i label importati vengono mescolati randomicamente
label_shuffle = False


def main():
    print("Load predictions...")
    prediction = load_prediction()
    label = load_label(shuffle=label_shuffle)
    print("Compute and save accuracies...")
    compute_and_save_accuracies(prediction, label)

'''
 Questa funzione importa gli array con le previsioni 
 ed effettua un reshape.
 
 Esempio da (375, 8, 1024, 512) a (3000, 1024, 512)
'''
def load_prediction():
    num_files = len(os.listdir(layer_dir))
    out = []
    for index in pb.progressbar(range(num_files)):
        filename = layer_name + "_prediction_" + str(index)+".npy"
        out.append(np.load(layer_dir + filename))

    out = np.asarray(out)
    shape = out.shape
    print(shape)
    out.shape = (shape[0]*shape[1], 512, 1024)
    return np.transpose(out)

'''
 Questa funzione importa i label.
 NB. per generare i label si può trovare il codice
     nel notebook colab NeuronPrediction_HiddenLayer
'''
def load_label(shuffle=False):
    res = np.load(label_dir + "label.npy")
    if shuffle:
        np.random.shuffle(res)
    return res


'''
 Implementazione dello pseudocodice per il calcolo delle accuratezze
 veridicando diversi threshold
'''
def compute_and_save_accuracies(predictions, label):
    label = np.asarray(label).astype(bool)
    accuracies = []

    for neuron in pb.progressbar(predictions):

        accuracies.append([])
        # I threshold sono stati calcolati per ogni singolo neurone.
        threshold = np.linspace(neuron.min(), neuron.max(), num=12)[1:11]

        for elem in neuron:
            # Per ogni elemento salviamo un dizionario con accuratezza, threshold e inversione
            # (con inv si verifica se è necessario invertire l'output per trovare l'accuratezza migliore)
            best = {'acc': 0, 'th': 0, 'inv': False}

            for t in threshold:
                inv = False
                elem_b = np.where(elem > t, True, False)
                acc_b = (label & elem_b) | (~label & ~elem_b)
                acc = acc_b.mean()

                if (1 - acc) > acc:
                    acc = 1 - acc
                    inv = True

                if acc > best["acc"]:
                    best["acc"] = acc
                    best["th"] = t
                    best["inv"] = inv

            accuracies[-1].append(best)

    # Salvo le accuratezze finali
    final_out = {'layer': layer_name, 'accuracies': accuracies}
    if label_shuffle:
        outF = open("../results_accuracies/E2_3000/" + layer_name + "_random.json", "x")
    else:
        outF = open("../results_accuracies/E2_3000/" + layer_name + "_accuracies.json", "x")
    json_out = json.dumps(final_out, indent=2)
    outF.write(json_out)
    outF.close()


if __name__ == "__main__":
    main()
