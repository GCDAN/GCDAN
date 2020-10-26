# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable

import numpy as np
import json
from collections import deque, Counter
import random
import pickle

def prepare_campus(dates, data_path = '../../data/campus_mobility/'):
    data = {}
    data['vid_list'] = {}
    data['uid_list'] = {}
    data['data_neural'] = {}

    valid_data = {}
    tot_trace = 0
    for date in dates:
        file_path = data_path + date + '.json'
        f = open(file_path)
        raw = json.load(f)
        f.close()
        '''
        time_str = date + ' 00:00:00'
        time_arr = time.strptime(time_str, '%y-%m-%d %H:%M:%S')
        time_base = int(time.mktime(time_arr))
        '''

        for mac, trajs in raw.items():
            if mac == '':
                continue
            valid_trajs = []
            if mac not in valid_data.keys():
                valid_data[mac] = []
            for id, traj in trajs.items():
                if len(traj) <= 4:
                    continue
                valid_trajs.append(traj)
            if len(valid_trajs) > 0:
                valid_data[mac].extend(valid_trajs)

    for mac, valid_trajs in valid_data.items():
        if len(valid_trajs) < 10:
            continue
        tot_trace += len(valid_trajs)
        if mac not in data['uid_list'].keys():
            l = len(data['uid_list'])
            data['uid_list'][mac] = [l, len(valid_trajs)]
            data['data_neural'][l] = {'sessions':{}, 'train':[], 'test':[]}


        uid = data['uid_list'][mac][0]
        inds = list(range(len(valid_trajs)))
        #random.shuffle(inds)
        n_test = max(1, int(0.25 * len(inds)))
        #n_train = len(inds) - n_test
        for i in range(len(inds)):
            trace = []
            for traj in valid_trajs[inds[i]]:
                loc = traj[0]
                st = int(traj[1])
                ed = int(traj[2])
                if loc not in data['vid_list'].keys():
                    l = len(data['vid_list'])
                    data['vid_list'][loc] = [l, 0]
                data['vid_list'][loc][1] += 1
                st = int(st/300)
                ed = int(ed/300)

                if st >= 288:
                    st = 287
                    print('error start time:' + traj[1])
                if ed >= 288:
                    ed = 287
                    print('error end time:' + traj[2])
                trace.append([data['vid_list'][loc][0], st, ed])

            data['data_neural'][uid]['sessions'][i] = trace
            if i < n_test:
                data['data_neural'][uid]['test'].append(i)
            else:
                data['data_neural'][uid]['train'].append(i)
    print('user num:' + str(len(data['uid_list'])))
    print('total trace:' + str(tot_trace))
    print('loc num:' + str(len(data['vid_list'])))
    save_path = data_path + 'input.json'
    f_out = open(save_path,'w')
    json.dump(data, f_out)
    f_out.close()
    return data

def prepare_gowalla(data_path = '../../data/gowalla.json'):
    data = {}
    data['vid_list'] = {}
    data['uid_list'] = {}
    data['data_neural'] = {}

    valid_data = {}
    tot_trace = 0
    f = open(data_path)
    raw = json.load(f)
    f.close()
    '''
    time_str = date + ' 00:00:00'
    time_arr = time.strptime(time_str, '%y-%m-%d %H:%M:%S')
    time_base = int(time.mktime(time_arr))
    '''

    for uid, trajs in raw.items():
        valid_trajs = []
        if uid not in valid_data.keys():
            valid_data[uid] = []
        for id, traj in trajs.items():
            if len(traj) <= 3:
                continue
            valid_trajs.append(traj)
        if len(valid_trajs) > 0:
            valid_data[uid].extend(valid_trajs)

    for ind, valid_trajs in valid_data.items():
        if len(valid_trajs) < 4:
            continue
        tot_trace += len(valid_trajs)
        if ind not in data['uid_list'].keys():
            l = len(data['uid_list'])
            data['uid_list'][ind] = [l, len(valid_trajs)]
            data['data_neural'][l] = {'sessions':{}, 'train':[], 'test':[]}


        uid = data['uid_list'][ind][0]
        inds = list(range(len(valid_trajs)))
        #random.shuffle(inds)
        n_test = max(1, int(0.25 * len(inds)))
        #n_train = len(inds) - n_test
        for i in range(len(inds)):
            trace = []
            for traj in valid_trajs[inds[i]]:
                #print(traj)
                loc = traj[0]
                st = int(traj[1])-1254326400
                ed = int(traj[1])-1254326400
            
                if loc not in data['vid_list'].keys():
                    l = len(data['vid_list'])
                    data['vid_list'][loc] = [l, 0]
                data['vid_list'][loc][1] += 1
                '''
                st = int(st/300)
                ed = int(ed/300)

                if st >= 288:
                    st = 287
                    print('error start time:' + traj[1])
                if ed >= 288:
                    ed = 287
                    print('error end time:' + traj[2])
                '''
                st = int(st/(86400/2))
                ed = int(st/(86400/2))
                trace.append([data['vid_list'][loc][0], st, ed])

            data['data_neural'][uid]['sessions'][i] = trace
            if i < n_test:
                data['data_neural'][uid]['test'].append(i)
            else:
                data['data_neural'][uid]['train'].append(i)
    print('user num:' + str(len(data['uid_list'])))
    print('total trace:' + str(tot_trace))
    print('loc num:' + str(len(data['vid_list'])))
    return data
class RnnParameterData(object):
    def __init__(self, loc_emb_size=512, uid_emb_size=16, tim_emb_size=16, hidden_size=500, epoch_max = 50,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 data_path='../../data/campus_mobility/', save_path='../results/', data_name='foursquare'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        if data_name == 'campus':
            dates = ['19-12-02','19-12-03','19-12-04','19-12-05','19-12-06','19-12-07','19-12-08']
            data = prepare_campus(dates, data_path)
            self.vid_list = data['vid_list']
            self.uid_list = data['uid_list']
            self.data_neural = data['data_neural']
            self.tim_size = 288
            self.loc_size = len(self.vid_list)
            self.uid_size = len(self.uid_list)
            self.loc_emb_size = loc_emb_size
            self.tim_emb_size = tim_emb_size
            self.uid_emb_size = uid_emb_size

            self.epoch = epoch_max
            self.dropout_p = dropout_p
            self.use_cuda = False
            self.lr = lr
            self.lr_step = lr_step
            self.lr_decay = lr_decay
            self.optim = optim
            self.L2 = L2
            self.clip = clip

        elif data_name == 'foursquare':
            f = open(self.data_path + self.data_name + '.pk', 'rb')
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
            f.close()
            #data = {}
            self.vid_list = data['vid_list']
            self.uid_list = data['uid_list']
            self.data_neural = data['data_neural']
            tot_trace = 0
            for user, user_data in self.data_neural.items():
                #print(user_data)
                tot_trace += len(user_data)
                for idx, session in user_data['sessions'].items():
                    for trace in session:
                        trace.append(trace[-1])
            #print(self.data_neural)
            print('user num:' + str(len(data['uid_list'])))
            print('total trace:' + str(tot_trace))
            print('loc num:' + str(len(data['vid_list'])))
            self.tim_size = 48
            self.loc_size = len(self.vid_list)
            self.uid_size = len(self.uid_list)
            self.loc_emb_size = loc_emb_size
            self.tim_emb_size = tim_emb_size
            self.uid_emb_size = uid_emb_size

            self.epoch = epoch_max
            self.dropout_p = dropout_p
            self.use_cuda = False
            self.lr = lr
            self.lr_step = lr_step
            self.lr_decay = lr_decay
            self.optim = optim
            self.L2 = L2
            self.clip = clip


        elif data_name == 'gowalla':
            data = prepare_gowalla(data_path+self.data_name+'.json')
            self.vid_list = data['vid_list']
            self.uid_list = data['uid_list']
            self.data_neural = data['data_neural']
            self.tim_size = 800
            self.loc_size = len(self.vid_list)
            self.uid_size = len(self.uid_list)
            self.loc_emb_size = loc_emb_size
            self.tim_emb_size = tim_emb_size
            self.uid_emb_size = uid_emb_size

            self.epoch = epoch_max
            self.dropout_p = dropout_p
            self.use_cuda = False
            self.lr = lr
            self.lr_step = lr_step
            self.lr_decay = lr_decay
            self.optim = optim
            self.L2 = L2
            self.clip = clip

        print('prepare data done...')

