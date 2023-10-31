import os
import json
import config as conf
from transformers import EncoderDecoderModel
if conf.model_type == 'joint':
    from transformers import EncoderDecoderJointModel
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer

def create_base_model(config):
    if conf.model_type == 'joint':
        return EncoderDecoderJointModel.from_encoder_decoder_pretrained(config.encoder_class, config.encoder_class, config.decoder_class)
    else:
        return EncoderDecoderModel.from_encoder_decoder_pretrained(config.encoder_class, config.decoder_class) #BartModel.from_pretrained('fabart-large')#
    
def create_tokenizer(config):
    return BertTokenizer.from_pretrained('bert-base-uncased')

def create_base_model_classify(config):
    return AutoModelForSequenceClassification.from_pretrained(config.tokenizer_class)
    
def create_tokenizer_classify(config):
    return AutoTokenizer.from_pretrained(config.base_class) 

def gen_ngram(sent, n=2):
    words = sent.split()
    ngrams = []
    for i, token in enumerate(words):
        if i<=len(words)-n:
            ngram = '-'.join(words[i:i+n])
            ngrams.append(ngram)
    return ngrams


def count_match(ref, dec, n=2):
    counts = 0.
    for d_word in dec:
        if d_word in ref:
            counts += 1
    return counts


def rouge_2(gold_sent, decode_sent):
    bigrams_ref = gen_ngram(gold_sent, 2)
    bigrams_dec = gen_ngram(decode_sent, 2)
    if len(bigrams_ref) == 0:
        recall = 0.
    else:
        recall = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_ref)
    if len(bigrams_dec) == 0:
        precision = 0.
    else:
        precision = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_dec)
    if recall+precision == 0:
        f1_score = 0.
    else:
        f1_score = 2*recall*precision/(recall+precision)
    return f1_score