import os
import ast

import json
from datetime import datetime

date = datetime.now().strftime("%H%M%S_%d%m%y")
directory = '../cyclomatic_complexity/code_dataset'

def main():
    data = read_data(directory, quantity=1400)
    save_data(data)


def read_data(data_folder, quantity=1000):
    data = []
    counter = [quantity, quantity]

    for filename in os.listdir(data_folder):
        if counter[0] == 0 and counter[1] == 0:
            break
        print("Lettura File " + filename)
        f = open(os.path.join(directory, filename), 'r')
        lines = f.readlines()

        for line in lines:
            tmp_dict = ast.literal_eval(line.strip())
            if tmp_dict['label'] == "Correct" and counter[0] > 0:
                data.append({"function": tmp_dict['function'], "label": tmp_dict['label']})
                counter[0] -= 1
            elif counter[1] > 0:
                data.append({"function": tmp_dict['function'], "label": tmp_dict['label']})
                counter[1] -= 1

    return data


def save_data(data_cc):
    uni_err_scr = 0
    print("\nSalvataggio dati...")
    os.mkdir('../dataset_reducedVM/' + date)
    f2 = open('../dataset_reducedVM/' + date + '/data.jsontxt', 'x')
    for dt in data_cc:
        try:
            f2.write(json.dumps(dt) + "\n")
        except UnicodeEncodeError:
            uni_err_scr += 1
            pass
    f2.close()
    print("Scrittura - Numero Codici: ", len(data_cc) - uni_err_scr)
    print("Scrittura - UnicodeEncodeError: ", uni_err_scr)


if __name__ == "__main__":
    main()

