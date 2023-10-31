import os
import random
import json
import numpy as np
import jsonlines
import random
import math
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
k = 5
if __name__ == "__main__":
    train_data = []
    with open('data/wikitrain.jsonl', "r+", encoding = 'utf-8') as f:
        for item in jsonlines.Reader(f):
            train_data.append(item)
    kfold = StratifiedKFold(n_splits = k)
    x_idx = np.zeros((len(train_data),2))
    y_idx = np.zeros(len(train_data))
    cnt = 0
    for train_index, dev_index in kfold.split(x_idx,y_idx):
        train_data_split = itemgetter(*train_index)(train_data)
        print(len(train_data_split))
        dev_data_split = itemgetter(*dev_index)(train_data)
        print(len(dev_data_split))
        cnt += 1
        train_fold_path = "train_fold_"+str(cnt)+".jsonl"
        train_out_fold_path = os.path.join('data', train_fold_path)
        with open(train_out_fold_path,'w', encoding = 'utf-8') as f_test:
            for item in train_data_split:
                f_test.write(json.dumps(item)+"\n")
        dev_fold_path = "dev_fold_"+str(cnt)+".jsonl"
        dev_out_fold_path = os.path.join('data', dev_fold_path)
        with open(dev_out_fold_path,'w', encoding = 'utf-8') as f_test:
            for item in dev_data_split:
                f_test.write(json.dumps(item)+"\n")
    pass