'''
 Questo script genera il dataset etichettato per complessità ciclomatica
'''

import os
import ast
import re

import numpy as np
import json
from radon.complexity import cc_visit
from collections import Counter
from datetime import datetime

date = datetime.now().strftime("%H%M%S_%d%m%y")
directory = 'code_dataset'

'''
 Lo script si compone in 3 parti.
 - lettura dati
 - selezionamento dati 
 - salvataggio nuovo dataset
'''


def main():
    data_cc = read_data(directory, includeAWE=False)
    data_cc = select_data(data_cc, label=np.arange(3, 13), quantity=200, binary=True)
    save_data(data_cc)


'''
 Questa funzione di utility verifica la presenza di except, with e assert nel codice
'''
def existAWE(function):
    # Sostituzione delle stringhe del codice con una stringa unica
    # (perchè nelle stringhe potrebbe trovarsi la parola ricercata)
    tmp = re.sub(r"(\"[^\"]*\")|(\'[^\']*\')", "\"template\"", function)

    # Ricerca delle parole nel codice
    if re.findall(r"(except )|(with )|(assert )", tmp):
        return True

    return False


'''
 Funzione che legge i dati dai dataset forniti nel repository github di colab
 e ne calcola la complessità ciclomatica
 
 NB. Includere o escludere i codici con assert, with e exept non è risultato 
     in particolari differenze al momento, quindi può ritenersi superfluo
'''
def read_data(data_folder, includeAWE=True):
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
                    if includeAWE or existAWE(tmp_dict['function']):
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


'''
 Questa funzione seleziona i dati in modo da ottenere un dataset bilanciato
 - data_cc, dataset con complessità ciclomatica già calcolata
 - label, array che indica quali complessità ciclomatiche selezionare
 - quantity, indica quanti codici salvare per ogni complessità selzionata in label
 - binary, se vero i label vengono modificati in modo da avere valore binario
'''
def select_data(data_cc, label=[1, 2], quantity=100, binary=True):
    sel_data = []
    counter = np.full(label.shape, quantity)
    bn = [0, 0]

    for dt in data_cc:
        if (dt['label'] in label) and counter[np.where(label == dt['label'])[0][0]] > 0:
            counter[np.where(label == dt['label'])[0][0]] -= 1
            if binary:
                dt['label'] = 0 if dt['label'] <= label[int(len(label) / 2) - 1] else 1
                bn[dt['label']] += 1
            sel_data.append(dt)

    tmp = np.full(label.shape, quantity)
    print("\nDati Selezionati: ", tmp - counter)
    if binary:
        print("Divisi in: ", bn)
    return sel_data


'''
 Questa funzione salva i dati su file
'''
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


if __name__ == "__main__":
    main()
