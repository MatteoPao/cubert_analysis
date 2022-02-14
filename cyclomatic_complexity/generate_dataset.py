import os
import ast
import numpy as np
import json
from radon.complexity import cc_visit
from collections import Counter
from datetime import datetime

date = datetime.now().strftime("%H%M%S_%d%m%y")
directory = 'code_dataset'


def read_data(data_folder):
    data_cc = []
    list_cc = []
    syn_err = 0
    uni_err = 0

    for filename in os.listdir(data_folder):
        print("Lettura File " + filename)
        f = open(os.path.join(directory, filename), 'r')
        lines = f.readlines()

        for line in lines:
            tmp_dict = ast.literal_eval(line.strip())
            if tmp_dict['label'] == "Correct":
                try:
                    #print(cc_visit(tmp_dict['function']))
                    #break
                    cc = cc_visit(tmp_dict['function'])[0].complexity
                    list_cc.append(cc)
                    data_cc.append({"function": tmp_dict['function'], "label": cc})
                except SyntaxError:
                    syn_err += 1
                    pass
                except UnicodeEncodeError:
                    uni_err += 1
                    pass
        f.close()

    print("Lettura - Numero Codici: ", len(data_cc))
    print("Lettura - SyntaxError: ", syn_err)
    print("Lettura - UnicodeEncodeError: ", uni_err)
    print(Counter(list_cc))

    return data_cc


def save_data(data_cc):
    uni_err_scr = 0
    print("\nSalvataggio dati...")
    os.mkdir('result/' + date)
    f2 = open('result/' + date + '/data_cc.jsontxt', 'x')
    for dt in data_cc:
        try:
            f2.write(json.dumps(dt) + "\n")
        except UnicodeEncodeError:
            uni_err_scr += 1
            pass
    f2.close()
    print("Scrittura - Numero Codici: ", len(data_cc) - uni_err_scr)
    print("Scrittura - UnicodeEncodeError: ", uni_err_scr)


def select_data(data_cc, label=10, quantity=1000, binary=False):
    sel_data = []
    counter = np.full(label, quantity)
    bn = [0, 0]

    for dt in data_cc:
        if dt['label'] <= label and counter[dt['label'] - 1] > 0:
            counter[dt['label'] - 1] -= 1
            if binary:
                dt['label'] = 0 if dt['label'] <= label/2 else 1
                bn[dt['label']] += 1
            sel_data.append(dt)

    tmp = np.full(label, quantity)
    print("\nDati Selezionati: ", tmp - counter)
    if binary:
        print("Divisi in: ", bn)
    return sel_data


def main():
    data_cc = read_data(directory)
    data_cc = select_data(data_cc, label=6, quantity=5000, binary=True)
    save_data(data_cc)


if __name__ == "__main__":
    main()
