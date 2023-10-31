import random
import json
import numpy as np
import jsonlines
sql_path = './data/wikidev.jsonl'
data_test = {}
with open(sql_path, "r+", encoding = 'utf-8') as f:
    for item in jsonlines.Reader(f):
        data_test[item['qid']] = item
log_path = './output/wikidev.jsonl_epoch_3.log'
table_err = {}
tot = 0
with open(log_path, "r+", encoding = 'utf-8') as f:
    lines = f.readlines()
    for line in lines:
        qid = line.split(':')[0]
        qid = int(qid)
        table_id  = data_test[qid]['table_id']
        tot += 1
        if table_id in table_err:
            cnt = table_err[table_id]
            table_err[table_id] = cnt + 1
        else:
            table_err[table_id] = 1
table_list = sorted(table_err.items(), key=lambda x:x[1], reverse=True)
print(table_list[0:10])
print(tot)
        