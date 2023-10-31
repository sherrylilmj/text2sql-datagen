#!/usr/bin/env python
# coding: utf-8
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
from preprocess import *
from sqlgraph_data import *
from aggregators import *
from model import *
import os
import utils
import transformers
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def compute_reward(target,pred,classifier_model,tokenizer,mode=None):
    reward = []
    if mode == 'rouge':
        for i in range(len(target)):
            score = utils.rouge_2(target[i],pred[i])
            reward.append(score)
    else:
        with torch.no_grad():
            model.to(conf.device)
            model.eval()
            for i in range(len(target)):
                tokenize_result = tokenizer.encode_plus(
                            target[i],
                            pred[i],
                            add_special_tokens=True,
                            max_length = 64,
                            pad_to_max_length=True
                )
                    
                input_ids = np.array(tokenize_result["input_ids"])
                input_mask = tokenize_result["attention_mask"]
                input_ids = torch.tensor(input_ids).to(conf.device).unsqueeze(0)
                input_mask = torch.tensor(input_mask).to(conf.device).unsqueeze(0)
                logits = classifier_model(input_ids = input_ids, attention_mask = input_mask)[0]
                score = torch.softmax(logits, dim=1)[0][0]
                reward.append(score)
    return torch.tensor(reward).to(conf.device)
# def compute_reward(value_list, predict_text):
#     reward = []
#     for batch_i in range(len(value_list)):
#         reward_i = 0.0
#         value_list_i = value_list[batch_i].split(' ')
#         for j in range(len(value_list_i)):
#             if j == "~":
#                 continue
#             if value_list_i[j] not in predict_text[batch_i]:
#                 reward_i += 1.0
#         reward_i += 1.0
#         reward.append(reward_i)
#     return torch.tensor(reward).to(conf.device)
def save_model(model, optimizer, fname="best_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, fname)
    
def load_saved_model(model_path, model, optimizer = None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load optimizer state to GPU. Reference: https://github.com/pytorch/pytorch/issues/2830#issuecomment-336031198
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

def train_joint(model,train_loader, dev_loader, num_epochs, criterion, optimizer, conf, scheduler=None, print_every=20, eval_every=200):
    batch_num = len(train_loader)
    model.train()
    
    running_avg_loss = 0
    best_bleu = 0.0
    for epochs in range(num_epochs):
        loss_sum = 0
        start_time = time.time()
        print("Epoch", epochs)
        classifier_model = utils.create_base_model_classify(conf).to(conf.device)
        tokenizer = utils.create_tokenizer_classify(conf)
        #classifier_optimizer = transformers.AdamW(classifier_model.parameters(), lr=conf.learning_rate)
        load_saved_model('best_model_all_958.pth', classifier_model)
        for (batch_idx, collate_output) in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
            labels = collate_output['labels'].to(torch.device("cuda:0"))
            input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
            label_mask = collate_output['label_mask'].to(torch.device("cuda:0"))
            value_ids = collate_output['value'].to(torch.device("cuda:0"))
            input_ids_text = collate_output['input_id_text'].to(torch.device("cuda:0"))
            labels_text = collate_output['labels_text'].to(torch.device("cuda:0"))
            input_mask_text = collate_output['input_mask_text'].to(torch.device("cuda:0"))
            label_mask_text = collate_output['label_mask_text'].to(torch.device("cuda:0"))
            value_ids_text = collate_output['value'].to(torch.device("cuda:0"))
            loss, preds, target,_,_, pred_score, value_list, preds_sample, seq_prob, seq_prob_sample = model(
                input_ids = input_ids,
                decoder_input_ids = input_ids, 
                input_mask = input_mask,
                decoder_mask = label_mask, 
                labels = labels, 
                train = True, 
                value_ids = value_ids, 
                rl = conf.rl, 
                encoder_type = 'sql')

            batch_size, n_steps, vocab_size = pred_score.size()
            # RL
            if conf.rl == True:
                if conf.my_transformer == True:
                    greedy_rewards = 1-compute_reward(target,preds,classifier_model,tokenizer) # B x S, 1
                    sample_rewards = 1-compute_reward(target,preds_sample,classifier_model,tokenizer) 
                    rl_loss = torch.mean((greedy_rewards - sample_rewards) * torch.sum(torch.log(seq_prob_sample + 1e-12)))
                    loss = (1-conf.rl_rate) * loss + conf.rl_rate * rl_loss #torch.sum(nll * all_rewards) # B
                else:
                    rewards = compute_reward(target,preds,classifier_model,tokenizer)
                    loss = torch.mean(loss*rewards)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
            optimizer.step()

            #text-to-text
            optimizer.zero_grad()
            model.train()
            text_loss, text_preds, text_target,_,_, text_pred_score, text_value_list, text_preds_sample, text_seq_prob, text_seq_prob_sample = model(
                input_ids = input_ids,
                decoder_input_ids = input_ids, 
                input_mask = input_mask,
                decoder_mask = label_mask, 
                labels = labels, 
                train = True, 
                value_ids = value_ids, 
                rl = conf.rl, 
                encoder_type = 'text')
            text_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf.max_grad_norm)
            optimizer.step()
            
            running_avg_loss += (loss.detach().item() + text_loss.detach().item())

            if (batch_idx > 0 and batch_idx % print_every == 0):
                msg =  "{}/{} - loss : {:.4f}" \
                        .format(batch_idx, batch_num, running_avg_loss/print_every)
                print(msg)
                running_avg_loss = 0
                
            if batch_idx % eval_every == eval_every-1:
                with torch.no_grad():
                    eval_avg_loss, bleu = evaluate(model, dev_loader, criterion, conf, verbose=False)
                    print('Eval Avg Loss: %f' % eval_avg_loss)
                    if (bleu > best_bleu):
                        best_bleu = bleu
                        save_model(model, optimizer,conf.save_model_path)

        end_time = time.time()
        print('Epoch Time:', end_time - start_time, 's')
        if scheduler is not None:
            scheduler.step()

def train(model, train_loader, dev_loader, num_epochs, criterion, optimizer, conf, scheduler=None, print_every=20, eval_every=200):
    batch_num = len(train_loader)
    model.train()
    
    running_avg_loss = 0
    best_bleu = 0.0
    for epochs in range(num_epochs):
        loss_sum = 0
        start_time = time.time()
        print("Epoch", epochs)
        classifier_model = utils.create_base_model_classify(conf).to(conf.device)
        tokenizer = utils.create_tokenizer_classify(conf)
        #classifier_optimizer = transformers.AdamW(classifier_model.parameters(), lr=conf.learning_rate)
        load_saved_model('best_model_all_958.pth', classifier_model)
        for (batch_idx, collate_output) in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
            labels = collate_output['labels'].to(torch.device("cuda:0"))
            input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
            label_mask = collate_output['label_mask'].to(torch.device("cuda:0"))
            value_ids = collate_output['value'].to(torch.device("cuda:0"))
            loss, preds, target,_,_, pred_score, value_list, preds_sample, seq_prob, seq_prob_sample = model(input_ids = input_ids,decoder_input_ids = input_ids, input_mask = input_mask,decoder_mask = label_mask, labels = labels, train = True, value_ids = value_ids, rl = conf.rl)
            batch_size, n_steps, vocab_size = pred_score.size()
            if conf.rl == True:
                if conf.my_transformer == True:
                    greedy_rewards = 1-compute_reward(target,preds,classifier_model,tokenizer) # B x S, 1
                    sample_rewards = 1-compute_reward(target,preds_sample,classifier_model,tokenizer) 
                    rl_loss = torch.mean((greedy_rewards - sample_rewards) * torch.sum(torch.log(seq_prob_sample + 1e-12)))
                    loss = (1-conf.rl_rate) * loss + conf.rl_rate * rl_loss #torch.sum(nll * all_rewards) # B
                else:
                    rewards = compute_reward(target,preds,classifier_model,tokenizer)
                    loss = torch.mean(loss*rewards)
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
                    eval_avg_loss, bleu = evaluate(model, dev_loader, criterion, conf, verbose=False)
                    print('Eval Avg Loss: %f' % eval_avg_loss)
                    if (bleu > best_bleu):
                        best_bleu = bleu
                        save_model(model, optimizer,conf.save_model_path)

        end_time = time.time()
        print('Epoch Time:', end_time - start_time, 's')
        if scheduler is not None:
            scheduler.step()

def evaluate(model, dev_loader, criterion,  conf, verbose=False):
    print("Start Eval")
    model.to(conf.device)
    model.eval()
    tokenizer = utils.create_tokenizer(conf)
    loss_sum = 0
    start_time = time.time()
    pds = []
    gts = []
    for (batch_idx, collate_output) in enumerate(dev_loader):

        input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
        labels = collate_output['labels'].to(torch.device("cuda:0"))
        input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
        label_mask = collate_output['label_mask'].to(torch.device("cuda:0"))
        loss, preds, target, generated_ids, target_ids,_,_,_,_,_ = model(input_ids, input_ids, input_mask, input_mask, labels, False, None, False)
        batch_size, nsteps = generated_ids.size()
        #print(labels.size())
        #print(pred.size())
        generated_ids = generated_ids[:,:nsteps].contiguous().view(batch_size*nsteps,-1)
        target_ids = target_ids[:,:nsteps].contiguous().view(batch_size*nsteps,-1)
        loss = loss.item()

        loss_sum += loss
        if verbose:
            print('Batch %d | Validation Loss %f' % (batch_idx, loss))
        
        for i in range(len(preds)):
            pd = preds[i]
            gt = target[i] 
            if i == 0 and batch_idx < 5:
                print(len(pd.split(' ')))
                print(len(gt.split(' ')))
                print("PD:", pd)
                print("GT:", gt)
            pds.append(pd.split(' '))
            gts.append([gt.split(' ')])
            if verbose and i == 0:
                print("PD:", pd)
                print("GT:", gt)

    bleu = bleu_score.corpus_bleu(gts, pds)
    avg_loss = loss_sum / len(dev_loader)
    end_time = time.time()
    print('Eval Time:', end_time - start_time, 's')
    print('Eval Avg Loss:', avg_loss)
    print('Dev BLEU-4:', bleu * 100)
    return avg_loss, bleu
def inference(model, test_loader, criterion,out_path, num_samples, verbose=False):
    print("Start Inference")
    model.to(conf.device)
    model.eval()

    start_time = time.time()
    pds = []
    out_file = open(out_path,'w')
    for (batch_idx, collate_output) in enumerate(test_loader):
        input_ids = collate_output['input_id'].to(torch.device("cuda:0"))
        labels = collate_output['labels'].to(torch.device("cuda:0"))
        input_mask = collate_output['input_mask'].to(torch.device("cuda:0"))
        label_mask = collate_output['label_mask'].to(torch.device("cuda:0"))

        loss, preds, target, generated_ids, target_ids,_,_,_,_,_ = model(input_ids, input_ids, input_mask, input_mask, labels, False)

        for i in range(len(preds)):
            gt = target[i//num_samples]
            pd = preds[i]
            out_file.write(pd+' <SEP> '+gt+"\n")
            if verbose:
                print("PD:", pd)
    out_file.close()
    end_time = time.time()
    print('Eval Time:', end_time - start_time, 's')
    #model.train()
    return 
if __name__ == "__main__":
    print("device:", conf.device)

    print("Loading data")
    if conf.model_type == 'joint':
        train_set = Joint_Dataset(conf.train_path, conf)
        dev_set = Joint_Dataset(conf.dev_path, conf)
        test_set = Joint_Dataset(conf.test_path, conf)
        train_loader = DataLoader(train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=collate, drop_last=True)
        dev_loader = DataLoader(dev_set, batch_size=conf.dev_batch_size, shuffle=False, collate_fn=collate, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)

        model = Seq2seq(conf).to(conf.device)

        num_epochs = 100
        criterion = nn.NLLLoss(ignore_index=0)
        optimizer = transformers.AdamW(model.parameters(), lr=conf.learning_rate)
        scheduler = transformers.get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=100,
                        num_training_steps=1000
                        )
        optimizer.zero_grad()
        load_saved_model('best_model_my_trans_joint_50.pth', model, optimizer)
        #train_joint(model, train_loader, dev_loader, num_epochs, criterion, optimizer, conf, scheduler=scheduler, print_every=200, eval_every=800)
        with torch.no_grad():
          inference(model, test_loader, criterion, './data/train_my_trans_joint_left_50.txt', 1, verbose=True)
    else:
        train_set = SQLGraph_Dataset(conf.train_path, conf)
        dev_set = SQLGraph_Dataset(conf.dev_path, conf)

        train_loader = DataLoader(train_set, batch_size=conf.train_batch_size, shuffle=True, collate_fn=collate, drop_last=True)
        dev_loader = DataLoader(dev_set, batch_size=conf.dev_batch_size, shuffle=False, collate_fn=collate, drop_last=True)
        test_set = SQLGraph_Dataset(conf.test_path, conf)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate, drop_last=False)
        model = Seq2seq(conf).to(conf.device)

        num_epochs = 100
        criterion = nn.NLLLoss(ignore_index=0)
        #num_train_steps = int(len(train_set) * int(num_epochs / conf.train_batch_size))
        optimizer = transformers.AdamW(model.parameters(), lr=conf.learning_rate)
        scheduler = transformers.get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=100,
                        num_training_steps=1000
                        )
        optimizer.zero_grad()
        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 35, 40, 45], gamma=0.1)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        load_saved_model('best_model_my_trans_joint_50.pth', model, optimizer)
        #train(model, train_loader, dev_loader, num_epochs, criterion, optimizer, conf, scheduler=scheduler, print_every=200, eval_every=800)

        # with torch.no_grad():
        #     evaluate(model, test_loader, criterion, conf, verbose=False)
        # #num_samples = 5
        with torch.no_grad():
          inference(model, test_loader, criterion, './data/train_my_trans_joint_left_50.txt', 1, verbose=True)
        #save_model(model, optimizer, "final_model.pth")

