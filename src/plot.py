import matplotlib.pyplot as plt
import os
import json
import numpy as np
from operator import itemgetter

result_path = "../results_accuracies/accuracies_1-18_FT/accuracies_E2/"
#result_path = "../results_accuracies/accuracies_1-6to12-17_FT/"


def main():
    acc = get_acc(-1)

    #box_plot(acc)
    bar_plot(acc)


def bar_plot(acc):
    fig2, ax2 = plt.subplots()
    ax2.set_title('Good neurons')

    res = [[], [], []]
    for lay in acc:
        over80 = 0
        over75 = 0
        over70 = 0

        while(lay[over70] >= 0.7):
            over70 += 1
            if(lay[over70] >= 0.75):
                over75 += 1
                if (lay[over70] >= 0.8):
                    over80 += 1

        res[0].append(over70)
        res[1].append(over75)
        res[2].append(over80)

    width = 0.5
    x = np.arange(5)
    rects1 = ax2.bar(x, res[0], width, label='over 0.70')
    #rects2 = ax2.bar(x, res[1], width, label='over 0.75')
    #rects3 = ax2.bar(x + width, res[2], width, label='over 0.80')

    ax2.set_xticks([1, 2, 3, 4, 5], [5, 10, 15, 20, 24])
    ax2.set_ylabel('Accuracy over 0.7')
    ax2.set_xlabel('Layer')
    ax2.bar_label(rects1, padding=1)
    #ax2.bar_label(rects2, padding=1)
    #ax2.bar_label(rects3, padding=1)

    plt.show()


def box_plot(accuracies):
    fig1, ax1 = plt.subplots()
    ax1.set_title('Layer Accuracies')
    ax1.boxplot(accuracies, labels=[5, 10, 15, 20, 24], showfliers=True)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Layer')

    plt.ylim([0.5, 1])
    plt.show()


def get_acc(num_elem):
    files = os.listdir(result_path)
    results = []
    for file in files:
        if "_accuracies" in file:
            data = get_data_from_file(result_path + file)
            data = np.array(data['accuracies']).flatten()
            data = sorted(data, key=itemgetter('acc'), reverse=True)
            results.append([d['acc'] for d in data[:num_elem]])

    return results


def get_data_from_file(filename):
    # Opening JSON file
    f = open(filename)
    return json.load(f)


if __name__ == "__main__":
    main()
