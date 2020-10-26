# coding: utf-8
from __future__ import print_function
from __future__ import division
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
import json
import time
import argparse
import numpy as np
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

from model import TrajTransformer
from utils import RnnParameterData

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def generate_input_transformer(data_neural, mode, max_len = 50, max_tgt = 50, batch_n = 50, candidate=None, graph = None):
    if candidate is None:
        candidate = list(data_neural.keys())
    batch_n = min(batch_n, len(candidate))
    batches = []
    for _ in range(batch_n):
        batches.append({'src_loc':[], 'src_st':[], 'src_ed':[], 'tgt_loc':[], 'tgt_st':[], 'tgt_ed':[], 'tgt_len':[], 'tgt_y':[], 'uid':[]})
    random.shuffle(candidate)
    for uid, u in enumerate(candidate):
        sessions = data_neural[u]['sessions']
        train_id = list(data_neural[u][mode])
        src_locs = []
        src_sts = []
        src_eds = []
        tgt_locs = []
        tgt_sts = []
        tgt_eds = []
        target_len = []
        tgt_ys = []
        random.shuffle(train_id)
        if mode == 'train':
            #first half as input, last half as output
            train_num = int((len(train_id) + 1) / 2)
            src_id = train_id[:train_num]
            tgt_id = train_id[train_num:]
        elif mode == 'test':
            src_id = data_neural[u]['train']
            tgt_id = train_id
        if len(src_id) == 0 or len(tgt_id) == 0:
            print('not enough trace')
            continue
        for c, i in enumerate(src_id):
            session = sessions[i]
            src_loc = [s[0]+1 for s in session]
            src_st = [s[1]+1 for s in session]
            src_ed = [s[2]+1 for s in session]
            if len(src_loc) >= max_len:
                continue
            for _ in range(max_len-len(src_loc)):
                src_loc.append(0)
                src_st.append(0)
                src_ed.append(0)
            src_loc = np.reshape(np.array(src_loc), (1, max_len))
            src_st = np.reshape(np.array(src_st), (1, max_len))
            src_ed = np.reshape(np.array(src_ed), (1, max_len))
            src_loc = Variable(torch.LongTensor(src_loc))
            src_st = Variable(torch.LongTensor(src_st))
            src_ed = Variable(torch.LongTensor(src_ed))
            src_locs.append(src_loc)
            src_sts.append(src_st)
            src_eds.append(src_ed)
        for c, i in enumerate(tgt_id):
            session = sessions[i]
            tgt_loc = [s[0]+1 for s in session]
            tgt_st = [s[1]+1 for s in session]
            tgt_ed = [s[2]+1 for s in session]
            tgt_y = Variable(torch.LongTensor(np.array([s[0]+1 for s in session[1:]])))
            if len(tgt_loc) >= max_tgt:
                continue
            tgt_ys.append(tgt_y)
            target_len.append(len(tgt_y))
            for _ in range(max_tgt-len(tgt_loc)):
                tgt_loc.append(0)
                tgt_st.append(0)
                tgt_ed.append(0)
            
            tgt_loc = np.reshape(np.array(tgt_loc), (1, max_tgt))
            tgt_st = np.reshape(np.array(tgt_st), (1, max_tgt))
            tgt_ed = np.reshape(np.array(tgt_ed), (1, max_tgt))
            tgt_loc = Variable(torch.LongTensor(tgt_loc))
            tgt_st = Variable(torch.LongTensor(tgt_st))
            tgt_ed = Variable(torch.LongTensor(tgt_ed))
            tgt_locs.append(tgt_loc)
            tgt_sts.append(tgt_st)
            tgt_eds.append(tgt_ed)
        #{'src_loc':[], 'src_st':[], 'src_ed':[], 'tgt_loc':[], 'tgt_st':[], 'tgt_ed':[], 'tgt_len':[], 'tgt_y':[]}
        if len(src_locs) < 1 or len(tgt_locs) < 1:
            continue
        src_locs = torch.cat(src_locs, dim = 0)
        src_sts = torch.cat(src_sts, dim = 0)
        src_eds = torch.cat(src_eds, dim = 0)
        tgt_locs = torch.cat(tgt_locs, dim = 0)
        tgt_sts = torch.cat(tgt_sts, dim = 0)
        tgt_eds = torch.cat(tgt_eds, dim = 0)
        batch_ind = uid % batch_n
        batches[batch_ind]['src_loc'].append(src_locs)
        batches[batch_ind]['src_st'].append(src_sts)
        batches[batch_ind]['src_ed'].append(src_eds)
        batches[batch_ind]['tgt_loc'].append(tgt_locs)
        batches[batch_ind]['tgt_st'].append(tgt_sts)
        batches[batch_ind]['tgt_ed'].append(tgt_eds)
        batches[batch_ind]['tgt_len'].append(target_len)
        batches[batch_ind]['tgt_y'].append(tgt_ys)
        batches[batch_ind]['uid'].append(u)
    return batches

def gen_graph(parameters, thresh = 5):
    n_loc = parameters.loc_size
    graph = np.zeros((n_loc+1,n_loc+1))
    data_neural = parameters.data_neural
    l = parameters.loc_size
    for uid in data_neural.keys():
        sessions = data_neural[uid]['sessions']
        train_id = list(data_neural[uid]['train'])
        for ind in train_id:
            session = sessions[ind]
            locs = [s[0]+1 for s in session]
            for j in range(0, len(locs)-1):
                edge = (max(locs[j],locs[j+1]), min(locs[j],locs[j+1]))
                graph[locs[j]][locs[j+1]] += 1
                graph[locs[j+1]][locs[j]] += 1
    return torch.FloatTensor(graph > thresh)

def compute_loss(scores, tgt_y, tgt_len, criterion):
    #print('in compute_loss:')
    tot_loss = 0
    pred_n = 0
    for i in range(len(tgt_len)):
        score = F.log_softmax(scores[i,:tgt_len[i],:], dim = -1)
        loss = criterion(score, tgt_y[i])
        tot_loss += loss
    #print(loss)
    return tot_loss, len(tgt_len)
'''
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc
'''
def get_acc(scores, tgt_y, tgt_len):
    #print('in get_acc:')
    acc = np.zeros((4, 1))
    for i in range(len(tgt_len)):
        score = F.log_softmax(scores[i,:tgt_len[i],:], dim=-1)
        #print(score.size())
        val, idxx = score.data.topk(10, dim = -1)
        #print(idxx.size())
        predx = idxx.numpy()
        target = tgt_y[i]
        target = target.data.numpy()
        #acc = np.zeros((3, 1))
        for j, p in enumerate(predx):
            t = target[j]
            if t in p[:10] and t > 0:
                acc[0] += 1
            if t in p[:5] and t > 0:
                acc[1] += 1
            if t == p[0] and t > 0:
                acc[2] += 1
            acc[3] += 1
    return acc
def run_transform(batches, mode, model, opt, criterion, graph = None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    loss_list = []
    acc_rec = np.zeros((4,1))
    for i, batch in enumerate(batches):
        #print('batch:'+str(i))
        opt.zero_grad()
        loc = batch['src_loc']
        st = batch['src_st']
        ed = batch['src_ed']
        tgt_loc = batch['tgt_loc']
        tgt_st = batch['tgt_st']
        tgt_ed = batch['tgt_ed']
        tgt_y = batch['tgt_y']
        tgt_len = batch['tgt_len']
        uid = batch['uid']
        loss = None
        tot_tgt = 0
        for j in range(len(loc)):
            #print(tgt_len[j])
            user_scores = model(loc[j], st[j], ed[j], tgt_loc[j], tgt_st[j], tgt_ed[j], tgt_len[j], uid[j])
            user_loss, tgt_l = compute_loss(user_scores, tgt_y[j], tgt_len[j], criterion)
            if loss is None:
                loss = user_loss
            else:
                loss += user_loss
            tot_tgt += tgt_l
            if mode == 'test':
                acc = get_acc(user_scores, tgt_y[j], tgt_len[j])
                acc_rec += acc
        loss = loss / tot_tgt
        if mode == 'train':
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss_list.append(loss.data.numpy())
        #break
    if mode == 'train':
        avg_loss = np.mean(loss_list, dtype=np.float64)
        return model, avg_loss
    elif mode == 'test':
        avg_loss = np.mean(loss_list, dtype=np.float64)
        return avg_loss, acc_rec
def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  tim_emb_size=args.tim_emb_size,
                                  dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2,
                                  optim=args.optim,
                                  clip=args.clip, epoch_max=args.epoch_max, 
                                  data_path=args.data_path, save_path=args.save_path)
    print('*' * 15 + 'start training...' + '*' * 15)
    g = gen_graph(parameters)
    model = TrajTransformer(parameters = parameters, graph = g)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3)
    lr = parameters.lr
    candidate = list(parameters.data_neural.keys())

    
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoint/'
    res = {'avg_loss':[], 'avg_acc':[]}
    best_acc = [0, 0, 0]
    best_epoch = [-1, -1, -1]
    if not os.path.exists(SAVE_PATH + tmp_path):
        os.mkdir(SAVE_PATH + tmp_path)
    for epoch in range(parameters.epoch):
        st = time.time()
        train_batches = generate_input_transformer(parameters.data_neural, 'train', candidate=candidate)
        test_batches = generate_input_transformer(parameters.data_neural, 'test', candidate=candidate)
        #print('generate finished...')
        model.train()
        model, avg_loss = run_transform(train_batches, 'train', model, optimizer, criterion)
        print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
        model.eval()
        test_loss, acc = run_transform(test_batches, 'test', model, optimizer, criterion)
        acc_10 = acc[0] / acc[3]
        acc_5 = acc[1] / acc[3]
        acc_1 = acc[2] / acc[3]
        acc_epoch = [acc_10, acc_5, acc_1]
        for i in range(3):
            if best_acc[i] < acc_epoch[i]:
                best_acc[i] = acc_epoch[i]
                best_epoch[i] = epoch
        res['avg_loss'].append(test_loss)
        res['avg_acc'].append(acc_1)
        print('==>Test Epoch:{:0>2d} Loss:{:.4f}'.format(epoch, test_loss))
        print('==>Test Acc:' + str((acc_10, acc_5, acc_1)))
        ed = time.time()
        print('epoch {} cost time: {}'.format(epoch, ed-st))
        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)
        avg_acc = acc_1
        scheduler.step(avg_acc)
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(res['avg_acc'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
            print('load epoch={} model state'.format(load_epoch))
        if lr <= 0.1 * 1e-5:
            break

    return res['avg_acc'][-1]

if __name__ == '__main__':
    seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=512, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=128, help="user id embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=16, help="time embeddings size")
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--data_name', type=str, default='foursquare')
    #parser.add_argument('--data_name', type=str, default='campus')
    parser.add_argument('--learning_rate', type=float, default = 5 * 1e-5)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--save_path', type=str, default='../results/')
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()

    ours_acc = run(args)
    print('ours_acc:' + str(ours_acc))
