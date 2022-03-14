import sys
import os
import json
import numpy as np
from operator import itemgetter


def main():
    files = os.listdir(result_path)
    for file in files:
        if "_accuracies" in file:
            print("\nAnalisi file: ", file)
            data = get_data_from_file(result_path + file)
            best_ind = get_best_inds(data['accuracies'])
            print(best_ind)
            print(len(best_ind))
            avg = get_best_avgs(data['accuracies'])
            print(avg)
            print(len(avg))
            sum = get_best_sum(data['accuracies'])
            print(sum)
            print(len(sum))


def get_data_from_file(filename):
    # Opening JSON file
    f = open(filename)
    return json.load(f)


def get_best_inds(accuracies, elem=10):
    print("Num neuroni: ", len(accuracies))
    best_inds = []
    for i, neuron in enumerate(accuracies):
        for j, ind in enumerate(neuron):
            if ind['acc'] >= 0.7:
                best_inds.append({'neuron':i,
                                  'ind':j,
                                  'acc':ind['acc'],
                                  'th':ind['th']})
    best_inds = sorted(best_inds, key=itemgetter('acc'), reverse=True)
    return best_inds[:elem]


def get_best_avgs(accuracies, elem=10):
    best_avgs = []
    for i, neuron in enumerate(accuracies):
        inds = [ind['acc'] for ind in neuron]
        mean = np.mean(inds)
        best_avgs.append({'neuron':i,
                          'avg':mean})
    best_avgs = sorted(best_avgs, key=itemgetter('avg'), reverse=True)
    return best_avgs[:elem]

def get_best_sum(accuracies, elem=10):
    best_sum = []
    for i, neuron in enumerate(accuracies):
        count = sum(1 for ind in neuron if ind['acc'] >= 0.7)
        best_sum.append({'neuron': i,
                          'over_0.7': count})
    best_sum = sorted(best_sum, key=itemgetter('over_0.7'), reverse=True)
    return best_sum[:elem]




if __name__ == "__main__":
    if len(sys.argv) == 2:
        result_path = sys.argv[1]
    main()
