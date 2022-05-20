import logging
import statistics
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from operator import itemgetter

#path = "../results_accuracies/accuracies_1-18_FT/accuracies_E2/"
path = "../results_accuracies/Final_1to16_FT_VM/"
path2 = "../results_accuracies/Final_1to16_FT_E2/"


def main():
    #acc = get_acc(-1, path)
    #acc2 = get_acc(-1, path2)

    #acc1_1 = get_acc(250000, path)
    #acc2_1 = get_acc(1000, path)
    #acc1_2 = get_acc(250000, path2)
    #acc2_2 = get_acc(1000, path2)

    #arr = get_acc_array(path)
    #arr2 = get_acc_array(path2)

    #scatter_plot(arr)
    #box_plot(acc, acc1, acc2)
    #box_plot_confr(acc2_1, acc2_2)
    #bar_plot(acc, acc2)

def scatter_plot(sel):

    fig, ax = plt.subplots(nrows=1, ncols=3, sharex='all')

    for a in range(3):
        x_8, y_8 = [], []
        x_75, y_75 = [], []
        x_7, y_7 = [], []
        for i in range(sel[a].shape[0]):
            for j in range(sel[a].shape[1]):
                if sel[a][i][j]['acc'] >= 0.8:
                    x_8.append(i)
                    y_8.append(j)
                if 0.75 <= sel[a][i][j]['acc'] < 0.8:
                    x_75.append(i)
                    y_75.append(j)
                if 0.7 <= sel[a][i][j]['acc'] < 0.75:
                    x_7.append(i)
                    y_7.append(j)

        ax[a].scatter(x_7, y_7, s=[0.7] * len(x_7), label='over 0.7')
        ax[a].scatter(x_75, y_75, s=[1.5] * len(x_75), label='over 0.75')
        ax[a].scatter(x_8, y_8, s=[3] * len(x_8), label='over 0.8')

    ax[0].set(xlim=(-10, 1033), xticks=np.arange(0, 1200, 200),
              ylim=(-10, 521), yticks=np.arange(0, 600, 100))
    ax[1].set(ylim=(-10, 521), yticks=np.arange(0, 600, 100))
    ax[2].set(ylim=(-10, 521), yticks=np.arange(0, 600, 100))

    ax[0].set_title('Layer 5')
    ax[1].set_title('Layer 10')
    ax[2].set_title('Layer 15')

    plt.legend(loc='lower right', scatterpoints=3, markerscale=3)
    plt.show()


def bar_plot(acc, acc2=None):
    fig2, ax2 = plt.subplots()
    res = [[], [], []]
    res2 = [[], [], []]

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

    if acc2:
        for lay in acc2:
            over80 = 0
            over75 = 0
            over70 = 0

            while (lay[over70] >= 0.7):
                over70 += 1
                if (lay[over70] >= 0.75):
                    over75 += 1
                    if (lay[over70] >= 0.8):
                        over80 += 1

            res2[0].append(over70)
            res2[1].append(over75)
            res2[2].append(over80)

    width = 0.5
    x = np.array([0, 2, 4, 6, 8])

    rects1, rects2, rects3 = None, None, None
    rects4, rects5, rects6 = None, None, None
    if acc2:
        width = 0.5
        rects1 = ax2.bar(x - width/1.7, res[0], width, label='acc > 0.70', edgecolor='black')
        rects2 = ax2.bar(x - width/1.7, res[1], width, label='acc > 0.75', edgecolor='black')
        rects3 = ax2.bar(x - width/1.7, res[2], width, label='acc > 0.80', edgecolor='black')
        print(rects1)
        rects4 = ax2.bar(x + width/1.7, res2[0], width, color='blue', edgecolor='black')
        rects5 = ax2.bar(x + width/1.7, res2[1], width, color='tab:red', edgecolor='black')
        rects6 = ax2.bar(x + width/1.7, res2[2], width, color='green', edgecolor='black')
    else:
        rects1 = ax2.bar(x, res[0], width, label='over 0.70')
        rects2 = ax2.bar(x, res[1], width, label='over 0.75')
        rects3 = ax2.bar(x, res[2], width, label='over 0.80')

    #rects1 = ax2.bar(x - width/2, res[0], width/2, label='over 0.70')
    #rects2 = ax2.bar(x, res[1], width/2, label='over 0.75')
    #rects3 = ax2.bar(x + width/2, res[2], width/2, label='over 0.80')
    #ax2.set_xticks([-0.125, 0.875, 1.75, 2.75, 3.75], [5, 10, 15, 20, 24])

    ax2.set_xticks(x, [5, 10, 15, 20, 24])
    ax2.set_xticklabels(['Layer 5', 'Layer 10', 'Layer 15', 'Layer 20', 'Layer 24'], rotation=45)
    ax2.set_ylabel('# of Neurons')
    ax2.bar_label(rects1, padding=2, fontsize=6)
    ax2.bar_label(rects4, padding=2, fontsize=6)
    #ax2.bar_label(rects3, padding=1)

    leg = ax2.legend(loc='upper center', title='D3')
    # Add second legend for the maxes and mins.
    # leg1 will be removed from figure
    leg2 = ax2.legend([rects4, rects5], ['acc > 0.70', 'acc > 0.75'], loc='upper right', title='D1')
    # Manually add the first legend back
    ax2.add_artist(leg)
    plt.show()


def box_plot(acc, acc1, acc2):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True)

    ax0.set_title('All neurons')
    ax0.boxplot(acc, labels=[5, 10, 15, 20, 24], showfliers=True)
    ax0.set_ylabel('Accuracy')
    ax0.set_ylim([0.5, 1])

    ax1.set_title('Best-half neurons')
    ax1.boxplot(acc1, labels=[5, 10, 15, 20, 24], showfliers=True)
    ax1.set_xlabel('Layer')
    ax1.set_ylim([0.5, 1])

    ax2.set_title('Best-1000 neurons')
    ax2.boxplot(acc2, labels=[5, 10, 15, 20, 24], showfliers=True)
    ax2.set_ylim([0.5, 1])

    plt.show()


def box_plot_confr(acc, acc1):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True)

    ax0.set_title('Best-1000 neurons D1-VM')
    ax0.boxplot(acc, showfliers=True)
    ax0.set_ylabel('Accuracy')
    ax0.set_ylim([0.5, 1])

    ax1.set_title('Best-1000 neurons D1-E2')
    ax1.boxplot(acc1, showfliers=True)

    ax1.set_xticklabels(['Layer 5', 'Layer 10', 'Layer 15', 'Layer 20', 'Layer 24', 'Layer 5', 'Layer 10', 'Layer 15', 'Layer 20', 'Layer 24'])
    ax1.set_ylim([0.5, 1])

    plt.show()

def get_acc(num_elem, result_path):
    files = os.listdir(result_path)
    results = []
    for file in files:
        if "_accuracies" in file:
            data = get_data_from_file(result_path + file)
            data = np.array(data['accuracies']).flatten()
            data = sorted(data, key=itemgetter('acc'), reverse=True)
            results.append([d['acc'] for d in data[:num_elem]])

    return results


def get_acc_array(result_path):
    files = os.listdir(result_path)
    results = []
    for file in files:
        if "_accuracies" in file:
            data = get_data_from_file(result_path + file)
            data = np.array(data['accuracies'])
            results.append(data)

    return results


def get_best_sum(accuracies):
    x, y = [], []
    for i, neuron in enumerate(accuracies):
        count = sum(1 for ind in neuron if ind['acc'] >= 0.7)
        x.append(i)
        y.append(count)

    return x, y


def get_data_from_file(filename):
    # Opening JSON file
    f = open(filename)
    return json.load(f)


if __name__ == "__main__":
    main()
