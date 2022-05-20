import os
import json
import numpy as np
import statistics
from operator import itemgetter


def get_acc_array(result_path):
    files = os.listdir(result_path)
    results = []
    for file in files:
        if "_accuracies" in file:
            data = get_data_from_file(result_path + file)
            data = np.array(data['accuracies'])
            results.append(data)

    return results


def get_data_from_file(filename):
    # Opening JSON file
    f = open(filename)
    return json.load(f)


path_1 = "../results_accuracies/Final_1to16_FT_E2/"
path_2 = "../results_accuracies/Final_1to16_FF_E2/"
path_3 = "../results_accuracies/Final_1-6to12-17_FT_E2/"

print("Caricamento ACC D1")
arr_1 = get_acc_array(path_1)
print("Caricamento ACC D2")
arr_2 = get_acc_array(path_2)
print("Caricamento ACC D3")
arr_3 = get_acc_array(path_3)


means = []
for i in range(1024):
    for j in range(512):
        mean = statistics.mean([arr_1[0][i][j]['acc'], arr_2[0][i][j]['acc'], arr_3[0][i][j]['acc']])
        mean_th = statistics.mean([arr_1[0][i][j]['th'], arr_2[0][i][j]['th'], arr_3[0][i][j]['th']])
        means.append({'mean': mean, 'mean_th': mean_th, 'x': i, 'y': j, 'orig': [arr_1[0][i][j], arr_2[0][i][j], arr_3[0][i][j]]})

means_sorted = sorted(means, key=itemgetter('mean'), reverse=True)

print(means_sorted[0])
print(means_sorted[1])
print(means_sorted[2])

x = 376
y = 282
print("\n--------------------------------------------")
print(path_1 + "\n Risultato: " + str(arr_1[0][x][y]))
print("\n--------------------------------------------")
print(path_2 + "\n Risultato: " + str(arr_2[0][x][y]))
print("\n--------------------------------------------")
print(path_3 + "\n Risultato: " + str(arr_3[0][x][y]))

