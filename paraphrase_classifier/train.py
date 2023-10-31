#!/usr/bin/env python
# coding: utf-8
import transformers
import numpy as np
import torch 
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from nltk.translate import bleu_score
import time

import config as conf
from data import *
import os
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def compute_reward(value_list, predict_text):
    reward = []
    for batch_i in range(len(value_list)):
        reward_i = 0.0
        value_list_i = value_list[batch_i].split(' ')
        for j in range(len(value_list_i)):
            if j == "~":
                continue
            if value_list_i[j] not in predict_text[batch_i]:
                reward_i += 1.0
        reward_i += 1.0
        reward.append(reward_i)
    return torch.tensor(reward).to(conf.device)
def save_model(model, optimizer, fname="best_model_all.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, fname)
    
def load_saved_model(model_path, model, optimizer):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
            
def idxs_to_sent(idxs, idx2word, oov_list=None):
    if oov_list is None:
        return ' '.join(list(map(lambda x : idx2word[str(x)], idxs)))
    else:
        tokens = []
        for i in idxs:
            if i < len(idx2word):
                tokens.append(idx2word[str(i)])
            else:
                i_oov = i - len(idx2word)
                tokens.append(oov_list[i_oov])
        return ' '.join(tokens)

def train(model, train_loader, dev_loader, num_epochs, optimizer, conf, scheduler=None, print_every=20, eval_every=200):
    batch_num = len(train_loader)
    model.train()
    
    running_avg_loss = 0
    best_acc = 0.0
    for epochs in range(num_epochs):
        loss_sum = 0
        start_time = time.time()
        print("Epoch", epochs)
        for (batch_idx, collate_output) in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
            labels = collate_output['labels'].to(torch.device("cuda:0"))
            input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
            loss = model(input_ids = input_ids, attention_mask = input_mask, labels = labels)[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
            optimizer.step()

            running_avg_loss += loss.detach().item()
            if (batch_idx > 0 and batch_idx % print_every == 0):
                msg =  "{}/{} - loss : {:.4f}" \
                        .format(batch_idx, batch_num, running_avg_loss/print_every)
                print(msg)
                running_avg_loss = 0
                
            if batch_idx % eval_every == eval_every-1:
                with torch.no_grad():
                    eval_avg_loss, eval_avg_acc = evaluate(model, dev_loader,  conf, verbose=False)
                    if (eval_avg_acc> best_acc):
                        best_acc = eval_avg_acc
                        save_model(model, optimizer, "best_model_all.pth")

        end_time = time.time()
        print('Epoch Time:', end_time - start_time, 's')
        if scheduler is not None:
            scheduler.step()
def compute_acc(results,labels):
    tot = 0
    cnt = 0
    for i in range(len(results)):
        if results[i] == labels[i]:
            cnt += 1.0
        tot += 1.0
    return (cnt/tot)
def evaluate(model, dev_loader,  conf, verbose=False):
    print("Start Eval")
    model.to(conf.device)
    model.eval()
    tokenizer = utils.create_tokenizer(conf)
    loss_sum = 0
    start_time = time.time()
    acc_sum = 0
    for (batch_idx, collate_output) in enumerate(dev_loader):

        input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
        labels = collate_output['labels'].to(torch.device("cuda:0"))
        input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
        loss, logits = model(input_ids = input_ids, attention_mask = input_mask, labels = labels)[:2]
        results = torch.softmax(logits, dim=1).argmax(axis=-1)
        acc = compute_acc(results.contiguous().view(-1).detach().cpu().numpy(), labels.contiguous().view(-1).detach().cpu().numpy())
        loss = loss.item()
        loss_sum += loss
        if verbose:
            print('Batch %d | Validation Loss %f' % (batch_idx, loss))
        acc_sum += acc
    avg_loss = loss_sum / len(dev_loader)
    avg_acc = acc_sum / len(dev_loader)
    end_time = time.time()
    print('Eval Time:', end_time - start_time, 's')
    print('Eval Avg Loss:', avg_loss)
    print('Eval Avg Accuracy:', avg_acc)
    return avg_loss, avg_acc
def inference(model, test_loader, criterion,out_path, num_samples, verbose=False):
    print("Start Inference")
    model.to(conf.device)
    model.eval()

    start_time = time.time()
    out_file = open(out_path,'w')
    for (batch_idx, collate_output) in enumerate(test_loader):
        input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
        labels = collate_output['labels'].to(torch.device("cuda:0"))
        input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))

        loss, logits = model(input_ids = input_ids, attention_mask = input_mask, labels = labels)[:2]
        results = torch.softmax(logits, dim=1).argmax(axis=-1)
        for i in range(len(results)):
            out_file.write(results[i])
            if verbose:
                print(results[i])
    out_file.close()
    end_time = time.time()
    print('Eval Time:', end_time - start_time, 's')
    return 
if __name__ == "__main__":
    print("device:", conf.device)

    print("Loading data")
    train_set = Para_Dataset(conf.train_path, conf)
    dev_set = Para_Dataset(conf.dev_path, conf)

    train_loader = DataLoader(train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    dev_loader = DataLoader(dev_set, batch_size=conf.dev_batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_set = Para_Dataset(conf.test_path, conf)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)
    model = utils.create_base_model(conf).to(conf.device)

    num_epochs = 100
    #num_train_steps = int(len(train_set) * int(num_epochs / conf.train_batch_size))
    optimizer = transformers.AdamW(model.parameters(), lr=conf.learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=100,
                num_training_steps=1000
                )
    optimizer.zero_grad()
    load_saved_model('best_model_paws_913.pth', model, optimizer)
    train(model, train_loader, dev_loader, num_epochs, optimizer, conf, scheduler=scheduler, print_every=200, eval_every=1500)

    with torch.no_grad():
        evaluate(model, dev_loader, conf, verbose=False)
    # num_samples = 5
    # with torch.no_grad():
    #    inference(model, test_loader, criterion, './data/train_paws_fine_tune_rl_left_50.txt', 1, verbose=True)
    #save_model(model, optimizer, "final_model.pth")

