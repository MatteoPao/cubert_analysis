'''
 Questo script genera il dataset etichettato per complessità ciclomatica
'''

import os
import ast
import re

import numpy as np
import json
import progressbar
from radon.complexity import cc_visit
from collections import Counter
from datetime import datetime

from cubert.full_cubert_tokenizer import FullCuBertTokenizer
from cubert import tokenizer_registry

date = datetime.now().strftime("%H%M%S_%d%m%y")
directory = 'code_dataset'


def main():
    """
     Lo script si compone in 3 parti.
     - lettura dati
     - selezionamento dati
     - salvataggio nuovo dataset
    """

    tokenizer = FullCuBertTokenizer(code_tokenizer_class=tokenizer_registry.TokenizerEnum.PYTHON.value,
                                    vocab_file="../src/fitness/cubert_fitness/cubert_pretrained_model_variablemisuse/vocab.txt")

    data_cc = read_data(directory, tokenizer, includeTrunc=False, includeAWE=True)
    data_cc = select_data(data_cc, label=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), quantity=150, binary=True)
    save_data(data_cc)
    save_features(data_cc, tokenizer)


def existAWE(function):
    """
     Questa funzione di utility verifica la presenza di except, with e assert nel codice
    """

    # Sostituzione delle stringhe del codice con una stringa unica
    # (perchè nelle stringhe potrebbe trovarsi la parola ricercata)
    tmp = re.sub(r"(\"[^\"]*\")|(\'[^\']*\')", "\"template\"", function)

    # Ricerca delle parole nel codice
    if re.findall(r"(except )|(except:)|(with )|(with:)|(assert )", tmp):
        return True

    return False


def isOverMaxLength(tokenizer, data, max_length):
    token = tokenizer.tokenize(data['function'])
    if len(token) > max_length:
        return True
    return False


def read_data(data_folder, tokenizer, includeTrunc=False, includeAWE=True):
    """
     Funzione che legge i dati dai dataset forniti nel repository github di colab
     e ne calcola la complessità ciclomatica

     NB. Includere o escludere i codici con assert, with e exept non è risultato
         in particolari differenze al momento, quindi può ritenersi superfluo
    """

    data_cc = []
    list_cc = []
    syn_err = 0
    uni_err = 0

    for filename in os.listdir(data_folder):
        print("Lettura File " + filename)
        f = open(os.path.join(directory, filename), 'r')
        lines = f.readlines()

        for line in progressbar.progressbar(lines):
            tmp_dict = ast.literal_eval(line.strip())
            if tmp_dict['label'] == "Correct":
                try:
                    if includeAWE or not existAWE(tmp_dict['function']):
                        if includeTrunc or not isOverMaxLength(tokenizer, tmp_dict, 512):
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


def select_data(data_cc, label=[1, 2], quantity=100, binary=True):
    """
     Questa funzione seleziona i dati in modo da ottenere un dataset bilanciato
     - data_cc, dataset con complessità ciclomatica già calcolata
     - label, array che indica quali complessità ciclomatiche selezionare
     - quantity, indica quanti codici salvare per ogni complessità selzionata in label
     - binary, se vero i label vengono modificati in modo da avere valore binario
    """

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


def save_data(data_cc):
    """
     Questa funzione salva i dati su file
    """

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


def save_features(data_cc, tokenizer, max_seq_len=512):
    len_list = []
    labels = []
    out_data = [[], [], []]
    for ex in data_cc:
        token = tokenizer.tokenize(ex['function'])
        len_list.append(len(token))
        labels.append(ex['label'])
        input_ids = tokenizer.convert_tokens_to_ids(token)

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            segment_ids.append(0)
            input_mask.append(0)

        out_data[0].append(input_ids)
        out_data[1].append(segment_ids)
        out_data[2].append(input_mask)

    print("Lista Lunghezze:")
    print(Counter(len_list))
    print("Lista Label:")
    print(Counter(labels))
    print("Shape output:")
    print(np.asarray(out_data).shape)

    with open('result/' + date + '/features.npy', 'wb') as file1:
        np.save(file1, out_data)

    with open('result/' + date + '/labels.npy', 'wb') as file2:
        np.save(file2, labels)


if __name__ == "__main__":
    main()


def read_data_old(data_folder, includeAWE=True):
    data_cc = []
    list_cc = []
    syn_err = 0
    uni_err = 0

    for filename in os.listdir(data_folder):
        print("Lettura File " + filename)
        f = open(os.path.join(directory, filename), 'r')
        lines = f.readlines()

        for line in progressbar.progressbar(lines):
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
