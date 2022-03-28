import sys
import os
import json
import numpy as np
from operator import itemgetter

result_path = "../results_accuracies/E2_2000/"

def main():
    files = os.listdir(result_path)
    result = []
    for file in files:
        if "_accuracies" in file:
            name = file.replace('_accuracies.json', '')
            print("\nAnalisi file: ", name)
            data = get_data_from_file(result_path + file)
            best_inds = get_best_inds(data['accuracies'], elem=5)
            print(best_inds)
            avgs = get_best_avgs(data['accuracies'], elem=5)
            print(avgs)
            sums = get_best_sum(data['accuracies'], elem=5)
            print(sums)

            result.append({'layer': name,
                         'best_inds': best_inds,
                         'neuron_avgs': avgs,
                         'neuron_over0.7': sums})
    save_data(result)

'''
    Funzione per importare le accuratezze dal file
'''
def get_data_from_file(filename):
    # Opening JSON file
    f = open(filename)
    return json.load(f)

'''
    Funzione per la ricerca dei migliori elementi generici
'''
def get_best_inds(accuracies, elem=10):
    print("Num neuroni: ", len(accuracies))
    best_inds = []
    for i, neuron in enumerate(accuracies):
        for j, ind in enumerate(neuron):
            if ind['acc'] >= 0.65:
                best_inds.append({'neuron':i,
                                  'ind':j,
                                  'acc':ind['acc'],
                                  'th':ind['th'],
                                  'inv':ind['inv']})
    best_inds = sorted(best_inds, key=itemgetter('acc'), reverse=True)
    return best_inds[:elem]

'''
    Funzione per la ricerca delle migliori medie per neurone
'''
def get_best_avgs(accuracies, elem=10):
    best_avgs = []
    for i, neuron in enumerate(accuracies):
        inds = [ind['acc'] for ind in neuron]
        mean = np.mean(inds)
        best_avgs.append({'neuron':i,
                          'avg':mean})
    best_avgs = sorted(best_avgs, key=itemgetter('avg'), reverse=True)
    return best_avgs[:elem]

'''
    Funzione per la ricerca dei neuroni con il maggior numero di elementi con accuratezza >0.7
'''
def get_best_sum(accuracies, elem=10):
    best_sum = []
    for i, neuron in enumerate(accuracies):
        count = sum(1 for ind in neuron if ind['acc'] >= 0.7)
        best_sum.append({'neuron': i,
                          'over_0.7': count})
    best_sum = sorted(best_sum, key=itemgetter('over_0.7'), reverse=True)
    return best_sum[:elem]


def save_data(data):

    outF = open(result_path + "layer_analysis.json", "w")
    json_out = json.dumps(data, indent=2)
    outF.write(json_out)
    outF.close()


if __name__ == "__main__":
    main()
