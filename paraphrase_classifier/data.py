import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import *
from collections import OrderedDict
import json
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
class Para_Dataset(Dataset):
    def get_feature(self, config, text, text_2):
        max_total_length = int(config.max_query_len)
        tokenize_result = self.tokenizer.encode_plus(
                    text,
                    text_2,
                    add_special_tokens=True,
                    max_length = max_total_length,
                    pad_to_max_length=True
            )

        input_ids = np.array(tokenize_result["input_ids"])
        input_mask = tokenize_result["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        assert len(input_ids) == max_total_length
        return input_ids, input_mask
    def __init__(self, input_path, config):
        self.tokenizer = utils.create_tokenizer(config)
        self.config = config
        self.input_ids = []
        self.input_mask = []
        self.labels = []
        if config.mode == 'paws':
            with open(input_path) as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter= '\t')
                for line in tsvreader:
                    if line[1] == 'sentence1':
                        continue
                    input_id, input_mask = self.get_feature(config, line[1], line[2])
                    self.input_ids.append(input_id)
                    self.input_mask.append(input_mask)
                    self.labels.append(int(line[3]))
        if config.mode == 'merged':
            with open(input_path) as f:
                for line in f.readlines():
                    line_list = line.strip().split('\t')
                    if len(line_list) > 3:
                        continue
                    input_id, input_mask = self.get_feature(config, line_list[0], line_list[1])
                    self.input_ids.append(input_id)
                    self.input_mask.append(input_mask)
                    self.labels.append(int(line_list[2]))

        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.input_mask[index]


# Returns:
def collate(batch_data):
    input_ids, labels, input_mask = zip(*batch_data)
    input_ids = list(input_ids)
    labels = list(labels)
    input_mask = list(input_mask)
    return {
        'input_id': torch.tensor(input_ids).to(conf.device),
        'labels': torch.LongTensor(labels).to(conf.device),
        'input_mask': torch.tensor(input_mask).to(conf.device),
    }