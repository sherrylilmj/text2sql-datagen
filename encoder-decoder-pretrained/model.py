import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
import config as conf
from aggregators import *
# from decoder import *
import sys
import utils
if sys.version > '3':
	from queue import PriorityQueue
	from queue import Queue
else:
    from Queue import PriorityQueue
    from Queue import Queue
import random


INF = 1e12

class Seq2seq(nn.Module):
    def __init__(self, config):
        super(Seq2seq, self).__init__()
        self.config = config
        self.base_model = utils.create_base_model(config)

        self.dropout = config.dropout
        self.beam_width = conf.beam_width
        self.decoder_type = conf.decoder_type
        self.seq_max_len = conf.seq_max_len
        self.teacher_forcing_prob = conf.teacher_forcing_prob
        self.tokenizer = utils.create_tokenizer(config)

    def _step(self,input_ids,labels,input_mask,decoder_mask, encoder_type = None):
        if labels == None:
            return 0.0
        y_ids = labels[:,:].contiguous()
        lm_labels = labels[:,:].clone()
        if encoder_type != None: # for joint learning model
            outputs = self.base_model(
                encoder_type = encoder_type,
                input_ids=input_ids,
                attention_mask = input_mask,
                decoder_input_ids = y_ids,
                decoder_attention_mask = decoder_mask,
                lm_labels=lm_labels,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask = input_mask,
                decoder_input_ids = y_ids,
                decoder_attention_mask = decoder_mask,
                lm_labels=lm_labels,
            )
        loss = outputs[0]
        return loss, outputs[1]
    def forward(self, input_ids,decoder_input_ids = None,input_mask = None,decoder_mask = None, labels = None, train = True, value_ids = None, rl = False, encoder_type = None):
        loss = None
        preds = None
        target = None
        generated_ids = None
        predict_score = None
        value_list = None
        seq_prob = None
        seq_prob_sample = None
        generated_ids_sample = None
        preds_sample = None
        if train:
            loss, predict_score = self._step(input_ids,labels,input_mask,decoder_mask, encoder_type)
            if rl == True:
                #greedy decode
                output = self.base_model.generate(
                input_ids = input_ids,
                attention_mask = input_mask,
                num_beams=1,
                #num_return_sequences = 5,
                max_length=40,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                #decoder_start_token_id=self.tokenizer.cls_token,
                decoder_start_token_id = self.tokenizer.cls_token_id,
                eos_token_id = self.tokenizer.pad_token_id
                )
                generated_ids = output[0]
                seq_prob = output[1]
                preds = [
                    self.tokenizer.decode(g, skip_special_tokens=True)
                    #self.tokenizer.decode(g)
                    for g in generated_ids
                ]
                target = [
                        self.tokenizer.decode(t, skip_special_tokens=True)
                        #self.tokenizer.decode(t)
                        for t in labels
                ]
                # random sample
                output_sample = self.base_model.generate(
                input_ids = input_ids,
                attention_mask = input_mask,
                num_beams=1,
                do_sample = True,
                max_length=40,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                #decoder_start_token_id=self.tokenizer.cls_token,
                decoder_start_token_id = self.tokenizer.cls_token_id,
                eos_token_id = self.tokenizer.pad_token_id
                )
                generated_ids_sample = output_sample[0]
                seq_prob_sample = output_sample[1]
                preds_sample = [
                    self.tokenizer.decode(g, skip_special_tokens=True)
                    #self.tokenizer.decode(g)
                    for g in generated_ids_sample
                ]

        else:
            loss,_ = self._step(input_ids,labels,input_mask,decoder_mask, encoder_type)
            if conf.my_transformer == True:
                output = self.base_model.generate(
                input_ids = input_ids,
                attention_mask = input_mask,
                num_beams=1,
                #num_return_sequences = 5,
                max_length=40,
                temperature=0.7,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True,
                #decoder_start_token_id=self.tokenizer.cls_token,
                decoder_start_token_id = self.tokenizer.cls_token_id,
                eos_token_id = self.tokenizer.pad_token_id
                )
                generated_ids = output[0]
                seq_prob = output[1]
            else:
                generated_ids = self.base_model.generate(
                input_ids = input_ids,
                attention_mask = input_mask,
                num_beams=1,
                #num_return_sequences = 5,
                max_length=40,
                temperature=0.7,
                repetition_penalty=1.5,
                length_penalty=1.0,
                early_stopping=True,
                #decoder_start_token_id=self.tokenizer.cls_token,
                decoder_start_token_id = self.tokenizer.cls_token_id,
                eos_token_id = self.tokenizer.pad_token_id
                )
            preds = [
                    self.tokenizer.decode(g, skip_special_tokens=True)
                    #self.tokenizer.decode(g)
                    for g in generated_ids
            ]
            target = [
                    self.tokenizer.decode(t, skip_special_tokens=True)
                    #self.tokenizer.decode(t)
                    for t in labels
            ]
        if value_ids == None:
            value_list = []
        else:
            value_list = [
                self.tokenizer.decode(v, skip_special_tokens=True)
                #self.tokenizer.decode(g)
                for v in value_ids
            ]
        return loss, preds, target, generated_ids, labels, predict_score, value_list, preds_sample, seq_prob, seq_prob_sample