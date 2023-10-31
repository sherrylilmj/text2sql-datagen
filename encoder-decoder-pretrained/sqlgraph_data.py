import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import *
from collections import OrderedDict
import json
from preprocess import word_tokenize
import config as conf
from operator import itemgetter
import unicodedata
import utils
import csv
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\n" or c == "\r":
        return True
    cat = unicodedata.category(c)
    if cat == "Zs":
        return True
    return False


def is_punctuation(c):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(c)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(c)
    if cat.startswith("P") or cat.startswith("S"):
        return True
    return False


def basic_tokenize(doc):
    doc_tokens = []
    char_to_word = []
    word_to_char_start = []
    prev_is_whitespace = True
    prev_is_punc = False
    prev_is_num = False
    for pos, c in enumerate(doc):
        if is_whitespace(c):
            prev_is_whitespace = True
            prev_is_punc = False
        else:
            if prev_is_whitespace or is_punctuation(c) or prev_is_punc or (prev_is_num and not str(c).isnumeric()):
                doc_tokens.append(c)
                word_to_char_start.append(pos)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_punc = is_punctuation(c)
            prev_is_num = str(c).isnumeric()
        char_to_word.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word, word_to_char_start
class SQLGraph_Dataset(Dataset):
    def get_feature(self, config, text, sql=None):
        max_total_length = int(config.max_query_len)
        tokens_final, input_ids_final = [], [], 
        tokens = []
        #print(text)
        #print(text)
        query_tokens, _, _ = basic_tokenize(text)
        for i, query_token in enumerate(query_tokens):
            sub_tokens = self.tokenizer.tokenize(query_token)
            if len(sub_tokens) > 0:
                tokens.extend(sub_tokens)
        if sql!=None:
            sql_tokens = []
            sql_token, _, _ = basic_tokenize(sql)
            for i, query_token in enumerate(sql_token):
                sub_tokens = self.tokenizer.tokenize(query_token)
                if len(sub_tokens) > 0:
                    sql_tokens.extend(sub_tokens)
            tokenize_result = self.tokenizer.encode_plus(
                sql_tokens,
                truncation=True,
                max_length=max_total_length,
                truncation_strategy="longest_first",
                pad_to_max_length=True,
                add_special_tokens = True,
            )
            len_idx = 0
            for idx in range(len(tokenize_result['input_ids'])):
                if tokenize_result['input_ids'][idx] == 102:
                    len_idx = idx
                    break
            #print(len_idx)
            sql_ids = tokenize_result['input_ids'][:len_idx]
            sql_mask = tokenize_result['attention_mask'][:len_idx]
            #print(tokenize_result["input_ids"])
            tokenize_result = self.tokenizer.encode_plus(
                    tokens,
                    truncation=True,
                    max_length=max_total_length,
                    truncation_strategy="longest_first",
                    pad_to_max_length=True,
                    add_special_tokens = True,
                )
            text_ids = tokenize_result['input_ids'][1:]
            text_mask = tokenize_result['attention_mask'][1:]
            input_ids = sql_ids + text_ids
            input_ids = np.array(input_ids[:64])
            input_mask = sql_mask + text_mask
            input_mask = input_mask[:64]
            #print(tokenize_result["input_ids"])
        else:
            tokenize_result = self.tokenizer.encode_plus(
                    tokens,
                    max_length=max_total_length,
                    truncation=True,
                    truncation_strategy="longest_first",
                    pad_to_max_length=True,
                    add_special_tokens = True,
                )

            input_ids = np.array(tokenize_result["input_ids"])
            #segment_ids = tokenize_result["token_type_ids"]
            input_mask = tokenize_result["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_total_length
        return tokens,  input_ids, input_mask
    def __init__(self, input_path, config, data_type = None):
        self.tokenizer = utils.create_tokenizer(config)
        self.config = config
        self.input_ids = []
        self.input_mask = []
        self.labels = []
        self.label_mask = []
        self.value = []
        if config.model_type == 'text-to-text':
            with open(input_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    jo = json.loads(line, object_pairs_hook=OrderedDict)
                    _, input_id, input_mask = self.get_feature(config, jo['pseudo_nl'],None)#sql_str
                    if jo['text'] == "":
                        jo['text'] = "test"
                    _, label, label_mask = self.get_feature(config, jo['text'])
                    self.input_ids.append(input_id)
                    self.input_mask.append(input_mask)
                    self.labels.append(label)
                    self.label_mask.append(label_mask)
                    value_str = ""
                    for cond in jo['sql']['conds']:
                        value_str += (str(cond[2]) + " ")
                    value_str = value_str.strip()
                    if len(value_str) < 2:
                        value_str = "~"
                    _, value_id, _ = self.get_feature(config,value_str)
                    self.value.append(value_id)
        if config.model_type == 'sql-to-text':
            with open(input_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    #print(line)
                    jo = json.loads(line, object_pairs_hook=OrderedDict)
                    #print(jo)
                    sql_str = jo['sql_text']
                    sql_token_list = sql_str.split(' ')
                    #print(sql_str)
                    sql_str = ''
                    for sql_token in sql_token_list:
                        if sql_token == 'agg':
                            sql_str += ('# ')
                            continue
                        elif sql_token == 'column':
                            continue
                        elif sql_token == 'end':
                            continue
                        elif sql_token == 'condition':
                            continue
                        elif sql_token == 'operator':
                            continue
                        elif sql_token == 'where':
                            sql_str += ('# '+sql_token + ' ')
                        else:
                            sql_str += (sql_token + ' ')
                    sql_str = sql_str.strip()
                    _, input_id, input_mask = self.get_feature(config, sql_str,None)#_, input_id, input_mask = self.get_feature(config, jo['pseudo_nl'],None)#sql_str
                    _, label, label_mask = self.get_feature(config, jo['text'])
                    #print(sql_str,jo['text'])
                    self.input_ids.append(input_id)
                    self.input_mask.append(input_mask)
                    self.labels.append(label)
                    self.label_mask.append(label_mask)
                    value_str = ""
                    for cond in jo['sql']['conds']:
                        value_str += (str(cond[2]) + " ")
                    value_str = value_str.strip()
                    if len(value_str) < 2:
                        value_str = "~"
                    _, value_id, _ = self.get_feature(config,value_str)
                    self.value.append(value_id)
        if config.model_type == 'combined':
            with open(input_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    #print(line)
                    jo = json.loads(line, object_pairs_hook=OrderedDict)
                    #print(jo)
                    sql_str = jo['sql_text']
                    sql_token_list = sql_str.split(' ')
                    #print(sql_str)
                    sql_str = ''
                    for sql_token in sql_token_list:
                        if sql_token == 'agg':
                            sql_str += ('# ')
                            continue
                        elif sql_token == 'column':
                            continue
                        elif sql_token == 'end':
                            continue
                        elif sql_token == 'condition':
                            continue
                        elif sql_token == 'operator':
                            continue
                        elif sql_token == 'where':
                            sql_str += ('# '+sql_token + ' ')
                        else:
                            sql_str += (sql_token + ' ')
                    sql_str = sql_str.strip()
                    _, input_id, input_mask = self.get_feature(config, sql_str,None)##sql_str
                    _, label, label_mask = self.get_feature(config, jo['text'])
                    _, input_id_text, input_mask_text = self.get_feature(config, jo['pseudo_nl'],None)
                    #print(sql_str,jo['text'])
                    self.input_ids.append(input_id)
                    self.input_mask.append(input_mask)
                    self.input_ids.append(input_id_text)
                    self.input_mask.append(input_mask_text)
                    self.labels.append(label)
                    self.label_mask.append(label_mask)
                    self.labels.append(label)
                    self.label_mask.append(label_mask)
                    value_str = ""
                    for cond in jo['sql']['conds']:
                        value_str += (str(cond[2]) + " ")
                    value_str = value_str.strip()
                    if len(value_str) < 2:
                        value_str = "~"
                    _, value_id, _ = self.get_feature(config,value_str)
                    self.value.append(value_id)
                    self.value.append(value_id)
        # with open(input_path) as tsvfile:
        #     tsvreader = csv.reader(tsvfile, delimiter= '\t')
        #     for line in tsvreader:
        #         if line[3] == '1':
        #             # print(line[1])
        #             # print(line[2])
        #             _, input_id, input_mask = self.get_feature(config, line[1])
        #             _, label, label_mask = self.get_feature(config, line[2])
        #             self.input_ids.append(input_id)
        #             self.input_mask.append(input_mask)
        #             self.labels.append(label)
        #             self.label_mask.append(label_mask)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        #print(torch.tensor(self.input_ids[index]).long())
        return self.input_ids[index], self.labels[index], self.input_mask[index], self.label_mask[index], self.value[index]

class Joint_Dataset(Dataset):
    def get_feature(self, config, text, sql=None):
        max_total_length = int(config.max_query_len)
        tokens = []

        query_tokens, _, _ = basic_tokenize(text)
        for i, query_token in enumerate(query_tokens):
            sub_tokens = self.tokenizer.tokenize(query_token)
            if len(sub_tokens) > 0:
                tokens.extend(sub_tokens)

        tokenize_result = self.tokenizer.encode_plus(
                tokens,
                max_length=max_total_length,
                truncation=True,
                truncation_strategy="longest_first",
                pad_to_max_length=True,
                add_special_tokens = True,
            )

        input_ids = np.array(tokenize_result["input_ids"])
        #segment_ids = tokenize_result["token_type_ids"]
        input_mask = tokenize_result["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        assert len(input_ids) == max_total_length
        return tokens,  input_ids, input_mask
    def __init__(self, input_path, config, data_type = None):
        self.tokenizer = utils.create_tokenizer(config)
        self.config = config
        self.input_ids = []
        self.input_mask = []
        self.labels = []
        self.label_mask = []
        self.value = []
        self.input_ids_text = []
        self.input_mask_text = []
        self.labels_text = []
        self.label_mask_text = []
        self.value_text = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                #print(line)
                jo = json.loads(line, object_pairs_hook=OrderedDict)
                #print(jo)
                sql_str = jo['sql_text']
                sql_token_list = sql_str.split(' ')
                #print(sql_str)
                sql_str = ''
                for sql_token in sql_token_list:
                    if sql_token == 'agg':
                        sql_str += ('# ')
                        continue
                    elif sql_token == 'column':
                        continue
                    elif sql_token == 'end':
                        continue
                    elif sql_token == 'condition':
                        continue
                    elif sql_token == 'operator':
                        continue
                    elif sql_token == 'where':
                        sql_str += ('# '+sql_token + ' ')
                    else:
                        sql_str += (sql_token + ' ')
                sql_str = sql_str.strip()
                _, input_id, input_mask = self.get_feature(config, sql_str)
                _, label, label_mask = self.get_feature(config, jo['text'])
                _, input_id_text, input_mask_text = self.get_feature(config, jo['pseudo_nl'],None)
                #print(sql_str,jo['text'])
                self.input_ids.append(input_id)
                self.input_mask.append(input_mask)
                self.input_ids_text.append(input_id_text)
                self.input_mask_text.append(input_mask_text)
                self.labels.append(label)
                self.label_mask.append(label_mask)
                self.labels_text.append(label)
                self.label_mask_text.append(label_mask)
                value_str = ""
                for cond in jo['sql']['conds']:
                    value_str += (str(cond[2]) + " ")
                value_str = value_str.strip()
                if len(value_str) < 2:
                    value_str = "~"
                _, value_id, _ = self.get_feature(config,value_str)
                self.value.append(value_id)
                self.value_text.append(value_id)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        #print(torch.tensor(self.input_ids[index]).long())
        return self.input_ids[index], self.labels[index], self.input_mask[index], self.label_mask[index], self.value[index], self.input_ids_text[index], self.labels_text[index], self.input_mask_text[index], self.label_mask_text[index], self.value_text[index]

# Returns:
# idx_seqs: NL question word idx sequence
# batch_g_ids: graph node ids for the batch (0-total number of nodes)
# batch_g_nodes: node ids for each data point (list of list)
# batch_features: node features
# batch_fw/bw_adj: adjacency lists
def collate(batch_data):
    if conf.model_type == 'joint':
        input_ids, labels, input_mask, label_mask, value, input_ids_text, labels_text, input_mask_text, label_mask_text, value_text = zip(*batch_data)
        input_ids = list(input_ids)
        labels = list(labels)
        input_mask = list(input_mask)
        label_mask = list(label_mask)
        value = list(value)
        input_ids_text = list(input_ids)
        labels_text = list(labels)
        input_mask_text = list(input_mask)
        label_mask_text = list(label_mask)
        value_text = list(value)
        return {
            'input_id': torch.tensor(input_ids).to(conf.device),
            'labels': torch.tensor(labels).to(conf.device),
            'input_mask': torch.tensor(input_mask).to(conf.device),
            'label_mask': torch.tensor(label_mask).to(conf.device),
            'value': torch.tensor(value).to(conf.device),
            'input_id_text': torch.tensor(input_ids).to(conf.device),
            'labels_text': torch.tensor(labels).to(conf.device),
            'input_mask_text': torch.tensor(input_mask).to(conf.device),
            'label_mask_text': torch.tensor(label_mask).to(conf.device),
            'value_text': torch.tensor(value).to(conf.device)
        }
    else:
        input_ids, labels, input_mask, label_mask, value = zip(*batch_data)
        input_ids = list(input_ids)
        labels = list(labels)
        input_mask = list(input_mask)
        label_mask = list(label_mask)
        value = list(value)
        return {
            'input_id': torch.tensor(input_ids).to(conf.device),
            'labels': torch.tensor(labels).to(conf.device),
            'input_mask': torch.tensor(input_mask).to(conf.device),
            'label_mask': torch.tensor(label_mask).to(conf.device),
            'value': torch.tensor(value).to(conf.device),
        }
