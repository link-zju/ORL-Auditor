'''
from asyncio import base_events
from concurrent.futures import process
from math import gamma
from time import process_time_ns, time
from turtle import position
import multiprocessing
from multiprocessing import Process, Manager
import gym
import pickle
import os 
import torch
import time
import copy
import torch.nn as nn
import numpy as np
from numpy.random import choice
import argparse
from scipy.special import softmax
from scipy import stats
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from torch.autograd import Variable

from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DQN, DiscreteSAC, SAC, DDPG
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from ndb import NDB 
'''

import re
from time import time
import math
import json
import pickle
from turtle import position
import multiprocessing
import random
from multiprocessing import Process, Manager
import os 
import torch
import time
import copy
import glob
import torch.nn as nn
import numpy as np
from numpy.random import choice
from scipy.special import softmax
from scipy import stats
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from .ndb import NDB
import gym 
import pandas as pd
from copy import deepcopy

import scipy
from scipy.stats import anderson
from ast import literal_eval

from d3rlpy.algos import DiscreteBC, BC, NFQ, DQN, DoubleDQN, DiscreteSAC, BCQ, DiscreteBCQ, DiscreteCQL, CQL, TD3PlusBC, IQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.wrappers.sb3 import to_mdp_dataset
from scipy.spatial.distance import cosine 
from scipy.stats import wasserstein_distance

def print_error(value):
    print("error: ", value)

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    # print(time_stamp)
    stamp = ("".join(time_stamp.split()[0].split("-"))+"".join(time_stamp.split()[1].split(":"))).replace('.', '')
    # print(stamp)

    return stamp

class CriticModel(nn.Module):
    def __init__(self, observation_action_size, n_hidden, n_output):
        super().__init__()
        self.fc1 = nn.Linear(observation_action_size, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 2*n_hidden)
        self.fc3 = nn.Linear(2*n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, obs_act):
        out = self.fc1(obs_act)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out = self.fc4(out)
        out = torch.sigmoid(out)
        out = torch.squeeze(out, dim=1)
        return out


class CriticModelWithoutLastActivation(CriticModel):

    def forward(self, obs_act):
        out = self.fc1(obs_act)
        out = F.relu(out)
        
        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
        out = F.relu(out)

        out = self.fc4(out)
        out = torch.squeeze(out, dim=1)
        return out


class MyAuditor():
    def __init__(self, 
                teacher_buffer_path, 
                num_of_cpu_processor, 
                num_of_gpu,
                train_epoch=200, 
                debug=True) -> None:

        self.teacher_buffer_path = teacher_buffer_path
        self.num_of_cpu_processor = num_of_cpu_processor
        self.num_of_gpu = num_of_gpu
        
        self.train_epoch = train_epoch
        self.debug = debug

    ## load agent from json
    @staticmethod
    def load_agent_from_json(json_path, suspected_model_type, cuda_id):
        if "CQL" == suspected_model_type:
            drl_agent = CQL.from_json(json_path, use_gpu=cuda_id)
        elif "DiscreteCQL" == suspected_model_type:
            drl_agent = DiscreteCQL.from_json(json_path, use_gpu=cuda_id)                    
            
        elif "DiscreteBC" == suspected_model_type:
            drl_agent = DiscreteBC.from_json(json_path, use_gpu=cuda_id)
        elif "BC" == suspected_model_type:
            drl_agent = BC.from_json(json_path, use_gpu=cuda_id)

        elif "DiscreteBCQ" == suspected_model_type:
            drl_agent = DiscreteBCQ.from_json(json_path, use_gpu=cuda_id)
        elif "BCQ" == suspected_model_type:
            drl_agent = BCQ.from_json(json_path, use_gpu=cuda_id)
                
        elif "DiscreteSAC" == suspected_model_type:
            drl_agent = DiscreteSAC.from_json(json_path, use_gpu=cuda_id)
        
        elif "TD3PlusBC" == suspected_model_type:
            drl_agent = TD3PlusBC.from_json(json_path, use_gpu=cuda_id)
            
        elif "IQL" == suspected_model_type:
            drl_agent = IQL.from_json(json_path, use_gpu=cuda_id)

        else: print("The suspected_model_type is illegal!")
        
        return drl_agent
    
    ## train the critic without the last activation layer
    @staticmethod
    def critic_generate(device, teacher_buffer_name, train_epoch=100, gamma=0.99):
        replay_dataset = np.load(teacher_buffer_name, allow_pickle=True)
        print('Replay_dataset\'s is numpy array')

        ## normailize the original reward to [-1, 1]
        max_abs_reward = max(abs(replay_dataset[:, -1]))
        replay_dataset[:, -1] = replay_dataset[:, -1] / max_abs_reward

        train_replay_dataset, test_replay_dataset = train_test_split(replay_dataset, test_size=0.2)

        ## train dataset initialization
        train_replay_dataset = torch.Tensor(train_replay_dataset)
        observation_action_size = int((train_replay_dataset.shape[-1] - 1) / 2)

        observation_action = train_replay_dataset[:, :observation_action_size]
        next_observation_action_and_reward = train_replay_dataset[:, observation_action_size:]
        # rewards = train_replay_dataset[:, 2*observation_action_size]
        train_dataset = TensorDataset(observation_action, next_observation_action_and_reward)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=2)

        ## test dataset initialization
        test_replay_dataset = torch.Tensor(test_replay_dataset)
        observation_action_size = int((test_replay_dataset.shape[-1] - 1) / 2)

        test_observation_action = test_replay_dataset[:, :observation_action_size]
        test_next_observation_action_and_reward = test_replay_dataset[:, observation_action_size:]
        # rewards = train_replay_dataset[:, 2*observation_action_size]
        test_dataset = TensorDataset(test_observation_action, test_next_observation_action_and_reward)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=True, num_workers=2)


        critic_model = CriticModelWithoutLastActivation(observation_action_size, n_hidden=1024, n_output=1).to(device)
        criterion = nn.SmoothL1Loss()
        
        # optimizer = torch.optim.SGD(critic_model.parameters(), lr=1e-3, momentum=0.9)
        # optimizer = torch.optim.Adam(critic_model.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(critic_model.parameters(), lr=1e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        for i in range(train_epoch):
            train_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                target_Q.detach()

                optimizer.zero_grad()
                Q = critic_model(observation_action)
                loss = criterion(Q, target_Q)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'[epoch: {i}] train loss: {train_loss:.3f}')
            
            if i%10 == 9:
                ## test loss
                with torch.no_grad():
                    test_loss = 0.0
                    for j, data in enumerate(testloader, 0):
                        observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                        next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                        target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                        target_Q.detach()

                        Q = critic_model(observation_action)
                        loss = criterion(Q, target_Q)
                        test_loss += loss.item()
                
                print(f'[epoch: {i}] test loss: {test_loss:.3f}')    
                
                ## save checkpoint
                save_path = teacher_buffer_name.replace('teacher_buffer_in_transition_form', 'trained_critic_model')
                save_path = save_path[:save_path.find('.')]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)    
                
                path = os.path.join(save_path, 'ckpt_{}.pt'.format(i+1))         
                torch.save({'epoch': i,
                            'model_state_dict': critic_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'test_loss': test_loss
                            },  path)
            ## learning rate decay
            scheduler.step()
                
        print('Finished Training')
           
   
    ## train the critic without the last activation layer
    @staticmethod
    def critic_generate_all_dataset(device, teacher_buffer_name, train_epoch=200, gamma=0.99):
        
        replay_dataset = np.load(teacher_buffer_name, allow_pickle=True)
        print('Replay_dataset\'s is numpy array')

        ## normailize the original reward to [-1, 1]
        max_abs_reward = max(abs(replay_dataset[:, -1]))
        replay_dataset[:, -1] = replay_dataset[:, -1] / max_abs_reward

        train_replay_dataset = replay_dataset

        ## train dataset initialization
        train_replay_dataset = torch.Tensor(train_replay_dataset)
        observation_action_size = int((train_replay_dataset.shape[-1] - 1) / 2)

        observation_action = train_replay_dataset[:, :observation_action_size]
        next_observation_action_and_reward = train_replay_dataset[:, observation_action_size:]
        # rewards = train_replay_dataset[:, 2*observation_action_size]
        
        train_data_ratio = 0.7
        train_data_length = int(train_data_ratio * observation_action.shape[0])
        index_arr = np.arange(observation_action.shape[0])
        np.random.shuffle(index_arr)
        
        
        train_dataset = TensorDataset(observation_action[index_arr[:train_data_length]], next_observation_action_and_reward[index_arr[:train_data_length]])
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=2)
        
        test_dataset = TensorDataset(observation_action[index_arr[train_data_length:]], next_observation_action_and_reward[index_arr[train_data_length:]])
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=True, num_workers=2)

        critic_model = CriticModelWithoutLastActivation(observation_action_size, n_hidden=1024, n_output=1).to(device)
        criterion = nn.SmoothL1Loss()
        
        # optimizer = torch.optim.SGD(critic_model.parameters(), lr=1e-3, momentum=0.9)
        # optimizer = torch.optim.Adam(critic_model.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(critic_model.parameters(), lr=1e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        for i in range(train_epoch):
            train_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                target_Q.detach()

                optimizer.zero_grad()
                Q = critic_model(observation_action)
                loss = criterion(Q, target_Q)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'[epoch: {i}] train loss: {train_loss:.3f}')
            
            with torch.no_grad():
                test_loss = 0
                for j, data in enumerate(testloader, 0):
                    observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                    next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                    target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                    target_Q.detach()
                    
                    Q = critic_model(observation_action)
                    loss = criterion(Q, target_Q)
                    # import pdb; pdb.set_trace()

                    test_loss += loss.item()
                    print(torch.mean(next_Q))
                print(f'[epoch: {i}] test loss: {test_loss:.3f}')
            
            if i%10 == 9: 
                ## save checkpoint
                save_path = teacher_buffer_name.replace('teacher_buffer_in_transition_form', 'trained_critic_model')
                save_path = save_path[:save_path.find('.h5')]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)    
                
                path = os.path.join(save_path, 'ckpt_{}.pt'.format(i+1))         
                torch.save({'epoch': i,
                            'model_state_dict': critic_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss
                            },  path)
            ## learning rate decay
            scheduler.step()
                
        print('Finished Training')
              

    ## retrain the critic without the last activation layer
    @staticmethod
    def critic_retrain(device, teacher_buffer_name, critic_model_name, train_epoch=900, gamma=0.99):
        critic_model_old = torch.load(critic_model_name)
        replay_dataset = np.load(teacher_buffer_name, allow_pickle=True)
        print('Replay_dataset\'s is numpy array')

        ## normailize the original reward to [-1, 1]
        max_abs_reward = max(abs(replay_dataset[:, -1]))
        replay_dataset[:, -1] = replay_dataset[:, -1] / max_abs_reward

        train_replay_dataset, test_replay_dataset = train_test_split(replay_dataset, test_size=0.2)

        ## train dataset initialization
        train_replay_dataset = torch.Tensor(train_replay_dataset)
        observation_action_size = int((train_replay_dataset.shape[-1] - 1) / 2)

        observation_action = train_replay_dataset[:, :observation_action_size]
        next_observation_action_and_reward = train_replay_dataset[:, observation_action_size:]
        # rewards = train_replay_dataset[:, 2*observation_action_size]
        train_dataset = TensorDataset(observation_action, next_observation_action_and_reward)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=2)

        ## test dataset initialization
        test_replay_dataset = torch.Tensor(test_replay_dataset)
        observation_action_size = int((test_replay_dataset.shape[-1] - 1) / 2)

        test_observation_action = test_replay_dataset[:, :observation_action_size]
        test_next_observation_action_and_reward = test_replay_dataset[:, observation_action_size:]
        # rewards = train_replay_dataset[:, 2*observation_action_size]
        test_dataset = TensorDataset(test_observation_action, test_next_observation_action_and_reward)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=True, num_workers=2)


        critic_model = CriticModelWithoutLastActivation(observation_action_size, n_hidden=1024, n_output=1).to(device)
        critic_model.load_state_dict(critic_model_old["model_state_dict"])
        criterion = nn.SmoothL1Loss()
        
        # optimizer = torch.optim.SGD(critic_model.parameters(), lr=1e-3, momentum=0.9)
        # optimizer = torch.optim.Adam(critic_model.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(critic_model.parameters(), lr=1e-4, amsgrad=True)
        optimizer.load_state_dict(critic_model_old["optimizer_state_dict"])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        for i in range(critic_model_old["epoch"]+1,critic_model_old["epoch"]+train_epoch+1, 1):
            train_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                target_Q.detach()

                optimizer.zero_grad()
                Q = critic_model(observation_action)
                loss = criterion(Q, target_Q)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            print(f'[epoch: {i}] train loss: {train_loss:.3f}')
            
            if i%10 == 9:
                ## test loss
                with torch.no_grad():
                    test_loss = 0.0
                    for j, data in enumerate(testloader, 0):
                        observation_action, next_observation_action_and_reward = data[0].to(device), data[1].to(device)

                        next_Q = critic_model(next_observation_action_and_reward[:, :-1])
                        target_Q = next_observation_action_and_reward[:, -1] + gamma * next_Q
                        target_Q.detach()

                        Q = critic_model(observation_action)
                        loss = criterion(Q, target_Q)
                        test_loss += loss.item()
                
                print(f'[epoch: {i}] test loss: {test_loss:.3f}')    
                
                ## save checkpoint
                save_path = teacher_buffer_name.replace('teacher_buffer_in_transition_form', 'trained_critic_model')
                save_path = save_path[:save_path.find('.')]

                if not os.path.exists(save_path):
                    os.makedirs(save_path)    
                
                path = os.path.join(save_path, 'ckpt_{}.pt'.format(i+1))         
                torch.save({'epoch': i,
                            'model_state_dict': critic_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'test_loss': test_loss
                            },  path)
            ## learning rate decay
            scheduler.step()
                
        print('Finished Training')
              
              
    @staticmethod
    def value_estimation(data_to_evaluate, critic_model_path, device):
        
        if type(data_to_evaluate).__module__ == np.__name__:
            states_and_actions = data_to_evaluate
        else:
            states_and_actions = np.load(data_to_evaluate, allow_pickle=True)

        observation_action_size = states_and_actions.shape[-1]
        
        # if 'rluply_d4rl_halfcheetah' in critic_model_path:
        #     n_hidden = 1024
        # elif 'sac_lunarlander' in critic_model_path:
        #     n_hidden = 128

        # elif 'sac_ant-v2' in critic_model_path:
        #     n_hidden = 1024
            
        # elif 'sac_bipedalwalker' in critic_model_path:
        #     n_hidden = 1024
            
        # else: print('Error n_hidden of critic_model')
        
        critic_model = CriticModelWithoutLastActivation(observation_action_size, n_hidden=1024, n_output=1).to(device)
        critic_model.load_state_dict(torch.load(critic_model_path, map_location=device)['model_state_dict'])
        critic_model.eval()
        
        states_and_actions = torch.Tensor(states_and_actions).to(device)
        critic_model_value = critic_model(states_and_actions)

        critic_model_value_arr = critic_model_value.cpu().detach().numpy()
        # print(max(critic_model_value_arr), min(critic_model_value_arr))
        # print(critic_model_value_arr[:20])
        return critic_model_value_arr
        # fingerprint_path = data_path_to_evaluate.replace('_states_actions.npy', '_states_actions_values.npy')
        # np.save(fingerprint_path, critic_model_value_arr)
        
    
    def fingerprint_sorted(self):
        ## load data
        # /home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/teacher_buffer_in_transition_form/baseline/model_SAC_20221017142406518-500000.h5.npy
        teacher_buffer_in_transition_form = np.load(self.teacher_buffer_name, allow_pickle=True)
        
        ## load model
        # /home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/teacher_buffer_in_transition_form/baseline/model_SAC_20221017142406518-500000.h5.npy
        critic_model_path = self.teacher_buffer_name.replace('teacher_buffer_in_transition_form', 'trained_critic_model')
        critic_model_path = critic_model_path[:critic_model_path.find('.')]

        critic_model_name = os.path.join(critic_model_path, self.critic_model_tag)

        observation_action_size = int((teacher_buffer_in_transition_form.shape[-1] - 1) / 2)
        
        critic_model = CriticModelWithoutLastActivation(observation_action_size, n_hidden=1024, n_output=1).to(self.device)
        critic_model.load_state_dict(torch.load(critic_model_name)['model_state_dict'])
        critic_model.eval()
        
        teacher_buffer_in_transition_form = torch.Tensor(teacher_buffer_in_transition_form).to(self.device)
        critic_model_value = critic_model(teacher_buffer_in_transition_form[:, :observation_action_size])

        sorted_model_value, sorted_value_indices = torch.sort(critic_model_value, descending=True, dim=-1)
        sorted_teacher_buffer_in_transition_form = teacher_buffer_in_transition_form[sorted_value_indices]
        fingerprint = sorted_teacher_buffer_in_transition_form[:, :observation_action_size]
        fingerprint_arr = fingerprint.cpu().detach().numpy()

        fingerprint_path = critic_model_path.replace('trained_critic_model', 'sorted_teacher_buffer_by_auditor')
        

        if not os.path.exists(fingerprint_path):
            os.makedirs(fingerprint_path)    
        fingerprint_name = 'fingerprint_from_critic_model-length_of_observation_{}-{}-{}.npy'.format(len(fingerprint), get_time_stamp(), self.critic_model_tag)
        path = os.path.join(fingerprint_path, fingerprint_name)
        np.save(path, fingerprint_arr)


    @staticmethod
    def data_audit(audit_dataset_path, critic_model_tag, suspected_model_path, result_save_path, suspected_model_tag, suspected_model_type, num_of_audited_episode, num_shadow_student, env_name, cuda_id, gamma=0.99):
        def feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode):    
        
            shadow_student_value_mean_list = []
            shadow_student_value_std_list = []
            
            shadow_student_value_mean_vertical_list = []
            shadow_student_value_std_vertical_list = []

            shadow_value_deviation_abs_sum_l1_list = []
            shadow_value_deviation_abs_sum_l2_list = []
            
            shadow_model_cos_distance_list = []
            shadow_model_cos_distance_weighted_list = []
            
            shadow_model_wasserstein_distance_list = []
            shadow_model_our_weighted_cosine_list = []
            
            model_list = []
            for shadow_student in shadow_student_list:
                json_path = shadow_student.replace(f"{suspected_model_tag}", "params.json")
                drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
                drl_shadow.load_model(shadow_student)
                model_list.append(copy.deepcopy(drl_shadow))
            
            for index_i, episode in enumerate(audit_mdpdataset):
                obs_data = episode.observations
                if index_i >= num_of_audited_episode:
                    break
                
                print(f"shadow model's episode: {index_i}")
                 
                shadow_student_value_list = []
                for drl_shadow in model_list:

                    shadow_actions = drl_shadow.predict(obs_data)
                    
                    shadow_actions = shadow_actions if len(shadow_actions.shape)>1 else np.expand_dims(shadow_actions, axis=1)

                
                    data_to_evaluate = np.concatenate((obs_data, shadow_actions), axis=1)
                    shadow_student_value = MyAuditor.value_estimation(data_to_evaluate, critic_model_path, device)
                    shadow_student_value_list.append(copy.deepcopy(shadow_student_value.reshape(-1, 1)))
                    
                shadow_student_value_stack = np.concatenate(shadow_student_value_list, axis=1)
                shadow_student_value_mean = np.mean(shadow_student_value_stack, axis=1)
                shadow_student_value_std = np.std(shadow_student_value_stack, axis=1)
                
                shadow_student_value_mean_vertical = np.mean(shadow_student_value_stack, axis=0)
                shadow_student_value_std_vertical = np.std(shadow_student_value_stack, axis=0)
                
                shadow_student_value_mean_list.append(shadow_student_value_mean)
                shadow_student_value_std_list.append(shadow_student_value_std)
                
                shadow_student_value_mean_vertical_list.append(shadow_student_value_mean_vertical)
                shadow_student_value_std_vertical_list.append(shadow_student_value_std_vertical)
            
                # the deviation between the mean value and the each shadow models' value
                shadow_student_value_mean = np.mean(shadow_student_value_stack, axis=1, keepdims=True)
                shadow_value_deviation_abs_sum = np.sum(np.abs(shadow_student_value_stack - shadow_student_value_mean), axis=0)
                shadow_value_deviation_abs_sum_l1_list.append(shadow_value_deviation_abs_sum)
                
                
                ## L2-norm
                shadow_value_deviation_abs_sum_l2 =  np.sum(np.square(shadow_student_value_stack - shadow_student_value_mean), axis=0)
                shadow_value_deviation_abs_sum_l2_list.append(shadow_value_deviation_abs_sum_l2)
                
                ## cos-similarity
                cosine_value_list = []
                normal_weighted_cosine_value_list = []
                wasserstein_distance_list = []
                our_weighted_cosine_value_list = []
                
                for i in range(shadow_student_value_stack.shape[1]):
                    cosine_value_list.append(cosine(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze()))
                    normal_weighted_cosine_value_list.append(cosine(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze(), np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze())))
                    wasserstein_distance_list.append(wasserstein_distance(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze()))
                    our_weighted_cosine_value_list.append(np.sum(np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze()) * shadow_student_value_stack[:, i].squeeze() * shadow_student_value_mean.squeeze())) 
                    
                cosine_value_arr = np.array(cosine_value_list)
                normal_weighted_cosine_value_arr = np.array(normal_weighted_cosine_value_list)
                wasserstein_distance_arr = np.array(wasserstein_distance_list)
                our_weighted_cosine_value_arr = np.array(our_weighted_cosine_value_list)
                
                shadow_model_cos_distance_list.append(cosine_value_arr)
                shadow_model_cos_distance_weighted_list.append(normal_weighted_cosine_value_arr)
                shadow_model_wasserstein_distance_list.append(wasserstein_distance_arr)
                shadow_model_our_weighted_cosine_list.append(our_weighted_cosine_value_arr)
                                
            
            return shadow_student_value_mean_list, shadow_student_value_std_list, shadow_student_value_mean_vertical_list, shadow_student_value_std_vertical_list, shadow_value_deviation_abs_sum_l1_list, shadow_value_deviation_abs_sum_l2_list, shadow_model_cos_distance_list, shadow_model_cos_distance_weighted_list, shadow_model_wasserstein_distance_list, shadow_model_our_weighted_cosine_list
        
        
        
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        audit_mdpdataset = MDPDataset.load(audit_dataset_path)
        audit_dataset_name = audit_dataset_path.split("/")[-1]
        

        
        critic_model_path = audit_dataset_path.replace("teacher_buffer", "auditor/trained_critic_model")
        critic_model_path = critic_model_path[:critic_model_path.find(".h5")]
        critic_model_path = os.path.join(critic_model_path, critic_model_tag)
        
        # teacher_of_student_list = os.listdir(suspected_model_path)
        teacher_of_student_list = glob.glob(f'{suspected_model_path}/*/{suspected_model_type}_*/*{suspected_model_tag}*')

        teacher_of_student_list = sorted(teacher_of_student_list)

        
        ## select shadow model
        shadow_student_list = []
        temp_teacher_of_student_list = copy.deepcopy(teacher_of_student_list)
        for teacher_of_student in temp_teacher_of_student_list:
            if audit_dataset_name in teacher_of_student and len(shadow_student_list) < num_shadow_student:        
                shadow_student_list.append(teacher_of_student)
                teacher_of_student_list.remove(teacher_of_student)
        
        shadow_student_value_mean_list, shadow_student_value_std_list, shadow_student_value_mean_vertical_list, shadow_student_value_std_vertical_list, shadow_value_deviation_abs_sum_l1_list, shadow_value_deviation_abs_sum_l2_list, shadow_model_cos_distance_list, shadow_model_cos_distance_weighted_list, shadow_model_wasserstein_distance_list, shadow_model_our_weighted_cosine_list = feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode)
        
        
        # data_record
        episode_record = []
        actual_buffer_name_record = []
        student_name_record = []
        audit_buffer_name_record = []
        teacher_student_l1_distance_record = []
        sum_teacher_teacher_next_record = []

        
        shadow_student_value_mean_vertical_record = []
        shadow_student_value_std_vertical_record = []
        suspected_model_value_mean_reward = []
        suspected_model_value_std_reward = []
        
        shadow_value_deviation_abs_sum_l1_record = []
        
        teacher_student_l2_distance_record = []
        shadow_value_deviation_abs_sum_l2_record = []
        
        teacher_student_cos_distance_record = []
        shadow_model_cos_distance_record = []
        
        teacher_student_cos_distance_weighted_record = []
        shadow_model_cos_distance_weighted_record = []
        
        teacher_student_wasserstein_distance_record = []
        shadow_model_wasserstein_distance_record = []
        
        teacher_student_our_weighted_cosine_record = []
        shadow_model_our_weighted_cosine_record = []
        
        for suspected_model_name in teacher_of_student_list:
            json_path = suspected_model_name.replace(f"{suspected_model_tag}", "params.json")
            drl_agent = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
            print(suspected_model_name)
            drl_agent.load_model(suspected_model_name)
            teacher_name = suspected_model_name.split("/")[-3]
            student_name = suspected_model_name.split("/")[-2]
            
            for index_i, episode in enumerate(audit_mdpdataset.episodes):
                if index_i >= num_of_audited_episode:
                    break
                print(f'episode: {index_i}')
                

                probe_actions1 = drl_agent.predict(episode.observations)                
                probe_actions1 = probe_actions1 if len(probe_actions1.shape)>1 else np.expand_dims(probe_actions1, axis=1)                    
                
                data_to_evaluate1 = np.concatenate((episode.observations, probe_actions1), axis=1)
                student1_data = MyAuditor.value_estimation(data_to_evaluate1, critic_model_path, device)
                
                
                ## L1-norm
                sum_teacher_student1 = sum(abs(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                teacher_student_l1_distance_record.append(sum_teacher_student1)
                shadow_value_deviation_abs_sum_l1_record.append(shadow_value_deviation_abs_sum_l1_list[index_i])
                
                
                ## L2-norm
                teacher_student_l2_distance = np.sum(np.square(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                teacher_student_l2_distance_record.append(teacher_student_l2_distance)
                shadow_value_deviation_abs_sum_l2_record.append(shadow_value_deviation_abs_sum_l2_list[index_i])
                
                ## cosine distance
                teacher_student_cos_distance_record.append(cosine(shadow_student_value_mean_list[index_i].squeeze(), student1_data.squeeze()))
                shadow_model_cos_distance_record.append(shadow_model_cos_distance_list[index_i])
                                     

                ## wasserstein distance
                teacher_student_wasserstein_distance_record.append(wasserstein_distance(student1_data.squeeze(), shadow_student_value_mean_list[index_i].squeeze()))
                shadow_model_wasserstein_distance_record.append(shadow_model_wasserstein_distance_list[index_i])
                
                
                episode_record.append(index_i)
                actual_buffer_name_record.append(teacher_name)
                student_name_record.append(student_name)
                audit_buffer_name_record.append(audit_dataset_name)
                
        data_dict = {'episode': episode_record, 
                    'actual_buffer_name': actual_buffer_name_record,
                    'student_name': student_name_record, 
                    'audit_buffer_name': audit_buffer_name_record, 
                    
                    'teacher_student_l1_distance': teacher_student_l1_distance_record,
                    'shadow_model_l1_distance': shadow_value_deviation_abs_sum_l1_record,
                    
                    'teacher_student_l2_distance': teacher_student_l2_distance_record,
                    'shadow_model_l2_distance': shadow_value_deviation_abs_sum_l2_record, 
                    
                    'teacher_student_cos_distance': teacher_student_cos_distance_record,
                    'shadow_model_cos_distance': shadow_model_cos_distance_record, 
                    
                    'teacher_student_wasserstein_distance': teacher_student_wasserstein_distance_record,
                    'shadow_model_wasserstein_distance': shadow_model_wasserstein_distance_record,     
                    }
        
        df = pd.DataFrame.from_dict(data_dict)
        
        df_save_path = os.path.join(result_save_path, 'audit_result-numepi_{}-audname_{}-critag_{}-sustype_{}-sustag_{}-numstu_{}-{}.xlsx'.format(num_of_audited_episode, audit_dataset_name, critic_model_tag[:critic_model_tag.find(".")], suspected_model_type, suspected_model_tag, num_shadow_student, get_time_stamp()))
        
        df.to_excel(df_save_path)
        
        MyAuditor.hypothesis_testing(df_save_path)
        
        
    
    @staticmethod
    def action_distance(audit_dataset_path, critic_model_tag, suspected_model_path, result_save_path, suspected_model_tag, suspected_model_type, num_of_audited_episode, num_shadow_student, env_name, cuda_id, gamma=0.99):
        def feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode):    
        
            shadow_actions_mean_value_list = []
            
            model_list = []
            for shadow_student in shadow_student_list:
                json_path = shadow_student.replace(f"{suspected_model_tag}", "params.json")
                drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
                drl_shadow.load_model(shadow_student)
                model_list.append(copy.deepcopy(drl_shadow))
            
            for index_i, episode in enumerate(audit_mdpdataset):
                obs_data = episode.observations
                if index_i >= num_of_audited_episode:
                    break
                
                print(f"shadow model's episode: {index_i}")
                 
                shadow_student_value_list = []
                for drl_shadow in model_list:
                    shadow_actions = drl_shadow.predict(obs_data)
                    
                    shadow_student_value = shadow_actions if len(shadow_actions.shape)>1 else np.expand_dims(shadow_actions, axis=1)
                    
                    shadow_student_value_list.append(copy.deepcopy(shadow_student_value.reshape(-1, 1)))
                    
                shadow_student_value_stack = np.concatenate(shadow_student_value_list, axis=1)
                shadow_actions = np.mean(shadow_student_value_stack, axis=1)        
                shadow_actions_mean_value_list.append(shadow_actions)                
            
            return shadow_actions_mean_value_list
        
        
        
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        audit_mdpdataset = MDPDataset.load(audit_dataset_path)
        audit_dataset_name = audit_dataset_path.split("/")[-1]
        

        
        critic_model_path = audit_dataset_path.replace("teacher_buffer", "auditor/trained_critic_model")
        critic_model_path = critic_model_path[:critic_model_path.find(".")]
        critic_model_path = os.path.join(critic_model_path, critic_model_tag)
        
        # teacher_of_student_list = os.listdir(suspected_model_path)
        teacher_of_student_list = glob.glob(f'{suspected_model_path}/*/{suspected_model_type}_*/*{suspected_model_tag}*')

        teacher_of_student_list = sorted(teacher_of_student_list)

        
        ## select shadow model
        shadow_student_list = []
        temp_teacher_of_student_list = copy.deepcopy(teacher_of_student_list)
        for teacher_of_student in temp_teacher_of_student_list:
            if audit_dataset_name in teacher_of_student and len(shadow_student_list) < num_shadow_student:        
                shadow_student_list.append(teacher_of_student)
                teacher_of_student_list.remove(teacher_of_student)
        
        shadow_student_value_mean_list = feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode)
        
        
        # data_record
        episode_record = []
        actual_buffer_name_record = []
        student_name_record = []
        audit_buffer_name_record = []
        
        teacher_student_l1_distance_record = []
        teacher_student_l2_distance_record = []
        teacher_student_cos_distance_record = []
        teacher_student_wasserstein_distance_record = []
        
        for suspected_model_name in teacher_of_student_list:
            json_path = suspected_model_name.replace(f"{suspected_model_tag}", "params.json")
            drl_agent = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
            print(suspected_model_name)
            drl_agent.load_model(suspected_model_name)
            teacher_name = suspected_model_name.split("/")[-3]
            student_name = suspected_model_name.split("/")[-2]
            
            for index_i, episode in enumerate(audit_mdpdataset.episodes):
                if index_i >= num_of_audited_episode:
                    break
                print(f'episode: {index_i}')
                
                # data_to_evaluate = np.concatenate((episode.observations, episode.actions), axis=1)
                # teacher_data = MyAuditor.value_estimation(data_to_evaluate, critic_model_path, device)
                # teacher_data_next = episode.rewards[:-1]/max_abs_reward + gamma * teacher_data[1:]

                probe_actions1 = drl_agent.predict(episode.observations)
                
                ## robustness test 20230130 by linkang
                # probe_actions1 = probe_actions1 + np.random.normal(0, noise_std, probe_actions1.shape)
                
                student1_data = probe_actions1 if len(probe_actions1.shape)>1 else np.expand_dims(probe_actions1, axis=1)       
                student1_data = student1_data.reshape(-1, 1)
                
                ## teacher_data
                sum_teacher_student1 = sum(abs(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                teacher_student_l1_distance_record.append(deepcopy(sum_teacher_student1))
                
                teacher_student_l2_distance = np.sum(np.square(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                teacher_student_l2_distance_record.append(deepcopy(teacher_student_l2_distance))
                
                teacher_student_cos_distance = cosine(shadow_student_value_mean_list[index_i].squeeze(), student1_data.squeeze())
                teacher_student_cos_distance_record.append(deepcopy(teacher_student_cos_distance))
                                
                ## wasserstein distance
                teacher_student_wasserstein_distance = wasserstein_distance(shadow_student_value_mean_list[index_i].squeeze(), student1_data.squeeze())
                teacher_student_wasserstein_distance_record.append(deepcopy(teacher_student_wasserstein_distance))

                episode_record.append(index_i)
                actual_buffer_name_record.append(teacher_name)
                student_name_record.append(student_name)
                audit_buffer_name_record.append(audit_dataset_name)

                
        data_dict = {'episode': episode_record, 
                    'actual_buffer_name': actual_buffer_name_record,
                    'student_name': student_name_record, 
                    'audit_buffer_name': audit_buffer_name_record, 
                    
                    'teacher_student_l1_distance': teacher_student_l1_distance_record,
        
                    'teacher_student_l2_distance': teacher_student_l2_distance_record,
                    
                    'teacher_student_cos_distance': teacher_student_cos_distance_record,
                    
                    'teacher_student_wasserstein_distance': teacher_student_wasserstein_distance_record
                    }
                    
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(result_save_path, 'action_distance-numepi_{}-audname_{}-critag_{}-sustype_{}-sustag_{}-numstu_{}-{}.csv'.format(num_of_audited_episode, audit_dataset_name, critic_model_tag[:critic_model_tag.find(".")], suspected_model_type, suspected_model_tag, num_shadow_student, get_time_stamp())))
        
        
        
    @staticmethod
    def state_action_value_record(audit_dataset_path, critic_model_tag, suspected_model_path, result_save_path, suspected_model_tag, suspected_model_type, num_of_audited_episode, num_shadow_student, env_name, cuda_id, gamma=0.99):
        
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        audit_mdpdataset = MDPDataset.load(audit_dataset_path)
        audit_dataset_name = audit_dataset_path.split("/")[-1]
        

        critic_model_path = audit_dataset_path.replace("teacher_buffer", "auditor/trained_critic_model")
        critic_model_path = critic_model_path[:critic_model_path.find(".")]
        critic_model_path = os.path.join(critic_model_path, critic_model_tag)
        
        # teacher_of_student_list = os.listdir(suspected_model_path)
        teacher_of_student_list = glob.glob(f'{suspected_model_path}/*/{suspected_model_type}_*/*{suspected_model_tag}*')

        teacher_of_student_list = sorted(teacher_of_student_list)

        
        ## select shadow model
        shadow_student_list = []
        temp_teacher_of_student_list = copy.deepcopy(teacher_of_student_list)
        for teacher_of_student in temp_teacher_of_student_list:
            if audit_dataset_name in teacher_of_student and len(shadow_student_list) < num_shadow_student:        
                shadow_student_list.append(teacher_of_student)
                teacher_of_student_list.remove(teacher_of_student)
        
        # shadow_student_value_mean_list, shadow_student_value_std_list, shadow_student_value_max_list, shadow_student_value_min_list, shadow_student_value_max_min_list, shadow_student_value_max_mean_distance_list, shadow_student_value_mean_vertical_list, shadow_student_value_std_vertical_list, shadow_value_deviation_abs_sum_l1_list, shadow_value_deviation_abs_sum_l2_list, shadow_model_cos_distance_list, shadow_model_cos_distance_weighted_list, shadow_model_wasserstein_distance_list, shadow_model_our_weighted_cosine_list = feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode)
        
        shadow_model_list = []
        for shadow_student in shadow_student_list:
            json_path = shadow_student.replace(f"{suspected_model_tag}", "params.json")
            drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
            drl_shadow.load_model(shadow_student)
            shadow_model_list.append(copy.deepcopy(drl_shadow))
            
        suspected_model_list = []
        for suspected_student in teacher_of_student_list:
            json_path = suspected_student.replace(f"{suspected_model_tag}", "params.json")
            drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
            drl_shadow.load_model(suspected_student)
            suspected_model_list.append(copy.deepcopy(drl_shadow))
            
        result_data = {}
        for index_i, episode in enumerate(audit_mdpdataset.episodes):
            episode_data = {}
            if index_i >= num_of_audited_episode:
                break
            print(f'episode: {index_i}')
            
            shadow_state_action_value_list = []
            for shadow_model in shadow_model_list:
                probe_actions1 = shadow_model.predict(episode.observations)
                probe_actions1 = probe_actions1 if len(probe_actions1.shape)>1 else np.expand_dims(probe_actions1, axis=1)                    
                data_to_evaluate1 = np.concatenate((episode.observations, probe_actions1), axis=1)
                student1_data = MyAuditor.value_estimation(data_to_evaluate1, critic_model_path, device)
                shadow_state_action_value_list.append(copy.deepcopy(student1_data.reshape(1, -1)))       
            shadow_value_stack = np.concatenate(shadow_state_action_value_list, axis=0)

            suspect_state_action_value_list = []
            for suspected_model in suspected_model_list:
                probe_actions2 = suspected_model.predict(episode.observations)
                probe_actions2 = probe_actions2 if len(probe_actions2.shape)>1 else np.expand_dims(probe_actions2, axis=1)                    
                data_to_evaluate1 = np.concatenate((episode.observations, probe_actions2), axis=1)
                student2_data = MyAuditor.value_estimation(data_to_evaluate1, critic_model_path, device)
                suspect_state_action_value_list.append(copy.deepcopy(student2_data.reshape(1, -1)))       
            suspect_value_stack = np.concatenate(suspect_state_action_value_list, axis=0)
            
            result_data[f"episode: {index_i}"] = {"shadow_value": shadow_value_stack,
                                                 "suspect_value": suspect_value_stack}
            
        # result_data["actual_dataset"] = teacher_name
        result_data["audit_dataset"] = audit_dataset_name
        result_data["num_shadow_model"] = len(shadow_model_list)
        result_data["num_suspected_model"] = len(suspected_model_list)
        
        save_name = os.path.join(result_save_path, 'state_action_values-numepi_{}-audname_{}-critag_{}-sustype_{}-sustag_{}-numstu_{}-{}.pk'.format(num_of_audited_episode, audit_dataset_name, critic_model_tag[:critic_model_tag.find(".")], suspected_model_type, suspected_model_tag, num_shadow_student, get_time_stamp()))
        
        with open(save_name, "wb") as write_file:
            pickle.dump(result_data, write_file)
        
    
    
    @staticmethod
    def data_audit_assemble_model(audit_dataset_path, critic_model_tag, suspected_model_path, result_save_path, suspected_model_tag, suspected_model_type, num_of_audited_episode, num_shadow_student, env_name, cuda_id, noise_std=0, gamma=0.99):
        def feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode):    
        
            shadow_student_value_mean_list = []
            shadow_student_value_std_list = []
            
            shadow_student_value_mean_vertical_list = []
            shadow_student_value_std_vertical_list = []
            
            shadow_student_value_max_list = []
            shadow_student_value_min_list = []
            shadow_student_value_max_min_list = []
            
            shadow_student_value_max_mean_distance_list = []
            shadow_value_deviation_abs_sum_l1_list = []
            shadow_value_deviation_abs_sum_l2_list = []
            
            shadow_model_cos_distance_list = []
            shadow_model_cos_distance_weighted_list = []
            
            shadow_model_wasserstein_distance_list = []
            shadow_model_our_weighted_cosine_list = []
            
            model_list = []
            for shadow_student in shadow_student_list:
                json_path = shadow_student.replace(f"{suspected_model_tag}", "params.json")
                drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
                drl_shadow.load_model(shadow_student)
                model_list.append(copy.deepcopy(drl_shadow))
            
            for index_i, episode in enumerate(audit_mdpdataset):
                obs_data = episode.observations
                if index_i >= num_of_audited_episode:
                    break
                
                print(f"shadow model's episode: {index_i}")
                 
                shadow_student_value_list = []
                for drl_shadow in model_list:
                    # json_path = shadow_student.replace(f"{suspected_model_tag}", "params.json")
                    # drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
                    # drl_shadow.load_model(shadow_student)
                    
                    
                    # import pdb; pdb.set_trace()
                    shadow_actions = drl_shadow.predict(obs_data)
                    
                    shadow_actions = shadow_actions if len(shadow_actions.shape)>1 else np.expand_dims(shadow_actions, axis=1)
                    # import pdb; pdb.set_trace()
                
                    data_to_evaluate = np.concatenate((obs_data, shadow_actions), axis=1)
                    shadow_student_value = MyAuditor.value_estimation(data_to_evaluate, critic_model_path, device)
                    shadow_student_value_list.append(copy.deepcopy(shadow_student_value.reshape(-1, 1)))
                    
                shadow_student_value_stack = np.concatenate(shadow_student_value_list, axis=1)
                shadow_student_value_mean = np.mean(shadow_student_value_stack, axis=1)
                shadow_student_value_std = np.std(shadow_student_value_stack, axis=1)
                
                shadow_student_value_mean_vertical = np.mean(shadow_student_value_stack, axis=0)
                shadow_student_value_std_vertical = np.std(shadow_student_value_stack, axis=0)
                
                shadow_student_value_mean_list.append(shadow_student_value_mean)
                shadow_student_value_std_list.append(shadow_student_value_std)
                
                shadow_student_value_mean_vertical_list.append(shadow_student_value_mean_vertical)
                shadow_student_value_std_vertical_list.append(shadow_student_value_std_vertical)
            
                ## max-min error
                # shadow_student_value_max = np.amax(shadow_student_value_stack, axis=1)
                # shadow_student_value_min = np.amin(shadow_student_value_stack, axis=1)
                # shadow_student_value_max_min = shadow_student_value_max - shadow_student_value_min
                # shadow_student_value_max_mean_distance = np.amax(np.abs(np.column_stack((shadow_student_value_max-shadow_student_value_mean, shadow_student_value_min-shadow_student_value_mean))), axis=1)
                
                # shadow_student_value_max_list.append(shadow_student_value_max)
                # shadow_student_value_min_list.append(shadow_student_value_min)
                # shadow_student_value_max_min_list.append(shadow_student_value_max_min)
                # shadow_student_value_max_mean_distance_list.append(shadow_student_value_max_mean_distance)
                # import pdb; pdb.set_trace()
                
                ## 202301222146
                # the deviation between the mean value and the each shadow models' value
                shadow_student_value_mean = np.mean(shadow_student_value_stack, axis=1, keepdims=True)
                shadow_value_deviation_abs_sum = np.sum(np.abs(shadow_student_value_stack - shadow_student_value_mean), axis=0)
                shadow_value_deviation_abs_sum_l1_list.append(shadow_value_deviation_abs_sum)
                
                
                ## 202301231119 L2-norm
                shadow_value_deviation_abs_sum_l2 =  np.sum(np.square(shadow_student_value_stack - shadow_student_value_mean), axis=0)
                shadow_value_deviation_abs_sum_l2_list.append(shadow_value_deviation_abs_sum_l2)
                
                ## 202301231250 cos-similarity
                cosine_value_list = []
                normal_weighted_cosine_value_list = []
                wasserstein_distance_list = []
                our_weighted_cosine_value_list = []
                
                for i in range(shadow_student_value_stack.shape[1]):
                    cosine_value_list.append(cosine(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze()))
                    normal_weighted_cosine_value_list.append(cosine(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze(), np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze())))
                    wasserstein_distance_list.append(wasserstein_distance(shadow_student_value_stack[:, i].squeeze(), shadow_student_value_mean.squeeze()))
                    our_weighted_cosine_value_list.append(np.sum(np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze()) * shadow_student_value_stack[:, i].squeeze() * shadow_student_value_mean.squeeze())) 
                    
                cosine_value_arr = np.array(cosine_value_list)
                normal_weighted_cosine_value_arr = np.array(normal_weighted_cosine_value_list)
                wasserstein_distance_arr = np.array(wasserstein_distance_list)
                our_weighted_cosine_value_arr = np.array(our_weighted_cosine_value_list)
                
                shadow_model_cos_distance_list.append(cosine_value_arr)
                shadow_model_cos_distance_weighted_list.append(normal_weighted_cosine_value_arr)
                shadow_model_wasserstein_distance_list.append(wasserstein_distance_arr)
                shadow_model_our_weighted_cosine_list.append(our_weighted_cosine_value_arr)
                
                # cosine_value_list = []
                # normal_weighted_cosine_value_list = []
                # wasserstein_distance_list = []
                # our_weighted_cosine_value_list = []
                                
                
                # ## 20230124 wasserstein distance
                # shadow_model_wasserstein_distance_list
                
                # cosine_value_list = []
                # for i in range(shadow_student_value_stack.shape[1]):
                #     # import pdb;pdb.set_trace()
                #     cosine_value_list.append(np.sum(np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze()) * shadow_student_value_stack[:, i].squeeze() * shadow_student_value_mean.squeeze())) 
                        
                # cosine_value_arr_weighted = np.array(cosine_value_list)
                # shadow_model_cos_distance_weighted_list.append(cosine_value_arr_weighted)
                
                
                
                # ## 202301231552 weighted cos-similarity
                # cosine_value_list = []
                # for i in range(shadow_student_value_stack.shape[1]):
                #     # import pdb;pdb.set_trace()
                #     cosine_value_list.append(np.sum(np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze()) * shadow_student_value_stack[:, i].squeeze() * shadow_student_value_mean.squeeze())) 
                        
                # cosine_value_arr_weighted = np.array(cosine_value_list)
                # shadow_model_cos_distance_weighted_list.append(cosine_value_arr_weighted)
                
            
            return shadow_student_value_mean_list, shadow_student_value_std_list, shadow_student_value_max_list, shadow_student_value_min_list, shadow_student_value_max_min_list, shadow_student_value_max_mean_distance_list, shadow_student_value_mean_vertical_list, shadow_student_value_std_vertical_list, shadow_value_deviation_abs_sum_l1_list, shadow_value_deviation_abs_sum_l2_list, shadow_model_cos_distance_list, shadow_model_cos_distance_weighted_list, shadow_model_wasserstein_distance_list, shadow_model_our_weighted_cosine_list
        
        
        
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        audit_mdpdataset = MDPDataset.load(audit_dataset_path)
        audit_dataset_name = audit_dataset_path.split("/")[-1]
        

        
        critic_model_path = audit_dataset_path.replace("teacher_buffer", "auditor/trained_critic_model")
        critic_model_path = critic_model_path[:critic_model_path.find(".")]
        critic_model_path = os.path.join(critic_model_path, critic_model_tag)
        
        # teacher_of_student_list = os.listdir(suspected_model_path)
        teacher_of_student_list = glob.glob(f'{suspected_model_path}/*/{suspected_model_type}_*/*{suspected_model_tag}*')

        teacher_of_student_list = sorted(teacher_of_student_list)

        
        ## select shadow model
        shadow_student_list = []
        temp_teacher_of_student_list = copy.deepcopy(teacher_of_student_list)
        for teacher_of_student in temp_teacher_of_student_list:
            if audit_dataset_name in teacher_of_student and len(shadow_student_list) < num_shadow_student:        
                shadow_student_list.append(teacher_of_student)
                teacher_of_student_list.remove(teacher_of_student)
        
        shadow_student_value_mean_list, shadow_student_value_std_list, shadow_student_value_max_list, shadow_student_value_min_list, shadow_student_value_max_min_list, shadow_student_value_max_mean_distance_list, shadow_student_value_mean_vertical_list, shadow_student_value_std_vertical_list, shadow_value_deviation_abs_sum_l1_list, shadow_value_deviation_abs_sum_l2_list, shadow_model_cos_distance_list, shadow_model_cos_distance_weighted_list, shadow_model_wasserstein_distance_list, shadow_model_our_weighted_cosine_list = feature_generator(shadow_student_list, critic_model_path, audit_mdpdataset, num_of_audited_episode)
        
        
        # data_record
        episode_record = []
        actual_buffer_name_record = []
        student_name_record = []
        audit_buffer_name_record = []
        teacher_student_l1_distance_record = []
        sum_teacher_student2_record = []
        sum_teacher_teacher_next_record = []

        mean_teacher_student1_record = []
        mean_teacher_student2_record = []
        mean_teacher_teacher_next_record = []

        var_teacher_student1_record = []
        var_teacher_student2_record = []
        var_teacher_teacher_next_record = []
        
        sum_max_min_record = []
        mean_max_min_record = []
        var_max_min_record = []
        
        sum_max_mean_distance_record = []
        mean_max_mean_distance_record = []
        var_max_mean_distance_record = []
        
        shadow_student_value_mean_vertical_record = []
        shadow_student_value_std_vertical_record = []
        suspected_model_value_mean_reward = []
        suspected_model_value_std_reward = []
        
        shadow_value_deviation_abs_sum_l1_record = []
        
        teacher_student_l2_distance_record = []
        shadow_value_deviation_abs_sum_l2_record = []
        
        teacher_student_cos_distance_record = []
        shadow_model_cos_distance_record = []
        
        teacher_student_cos_distance_weighted_record = []
        shadow_model_cos_distance_weighted_record = []
        
        teacher_student_wasserstein_distance_record = []
        shadow_model_wasserstein_distance_record = []
        
        teacher_student_our_weighted_cosine_record = []
        shadow_model_our_weighted_cosine_record = []
        
        
        # /home/linkang/my_projects/off-rl-proj/off-rl-data-nescgpu/fingerprint_sample_generate_double-check/sac_lunarlander/teacher_buffer/baseline/model_SAC_20221017142406518-500000.h5
        # /home/linkang/my_projects/off-rl-proj/off-rl-data-nescgpu/fingerprint_sample_generate_double-check/sac_lunarlander/student_policy/baseline/model_SAC_20221017142406518-500000.h5/BC_20230123134951
        
        # /home/linkang/my_projects/off-rl-proj/off-rl-data-nescgpu/fingerprint_sample_generate_double-check/sac_lunarlander/student_policy/overlap_split_5_indice_2/model_SAC_20221017142406518-500000-overlap_indice_2-subdataset_0.h5/BC_20230412174743
        # assemble model list
        assemble_expt_hyp = "overlap_split_5_indice_2"
        assemble_model_path = suspected_model_path.replace("baseline", assemble_expt_hyp)
        split_number = 5
        indice_number = 2
        assemble_model_list = glob.glob(f'{assemble_model_path}/*/{suspected_model_type}_*/*{suspected_model_tag}*')
        assemble_model_list = sorted(assemble_model_list)
        
        for suspected_model_index_i in range(0, len(assemble_model_list), split_number):
            drl_agent_list = []
            for suspected_model_index_j in range(indice_number):
                suspected_model_name = assemble_model_list[suspected_model_index_i + suspected_model_index_j + (indice_number-1)]
                json_path = suspected_model_name.replace(f"{suspected_model_tag}", "params.json")
                drl_agent = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=suspected_model_type, cuda_id=cuda_id)
                print(suspected_model_name)
                drl_agent.load_model(suspected_model_name)
                drl_agent_list.append(deepcopy(drl_agent))
                teacher_name = suspected_model_name.split("/")[-3]
                teacher_name = teacher_name[:teacher_name.find("-overlap_indice_2")]
                student_name = suspected_model_name.split("/")[-2]
                
            for index_i, episode in enumerate(audit_mdpdataset.episodes):
                if index_i >= num_of_audited_episode:
                    break
                print(f'episode: {index_i}')
                
                # data_to_evaluate = np.concatenate((episode.observations, episode.actions), axis=1)
                # teacher_data = MyAuditor.value_estimation(data_to_evaluate, critic_model_path, device)
                # teacher_data_next = episode.rewards[:-1]/max_abs_reward + gamma * teacher_data[1:]
                probe_actions1 = np.zeros_like(drl_agent_list[0].predict(episode.observations))
                for _drl_agent in drl_agent_list:
                    probe_actions1 += _drl_agent.predict(episode.observations)
                
                probe_actions1 = probe_actions1/len(drl_agent_list)
                
                ## robustness test 20230130 by linkang
                # probe_actions1 = probe_actions1 + np.random.normal(0, noise_std, probe_actions1.shape)
                
                probe_actions1 = probe_actions1 if len(probe_actions1.shape)>1 else np.expand_dims(probe_actions1, axis=1)                    
                
                data_to_evaluate1 = np.concatenate((episode.observations, probe_actions1), axis=1)
                student1_data = MyAuditor.value_estimation(data_to_evaluate1, critic_model_path, device)
                
                
                ## teacher_data
                sum_teacher_student1 = sum(abs(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                # mean_teacher_student1 = np.mean(abs(shadow_student_value_mean_list[index_i] - student1_data))
                # var_teacher_student1 = np.var(abs(shadow_student_value_mean_list[index_i] - student1_data))
                teacher_student_l1_distance_record.append(sum_teacher_student1)
                # mean_teacher_student1_record.append(mean_teacher_student1)
                # var_teacher_student1_record.append(var_teacher_student1)                                
                
                sum_teacher_teacher_next = sum(shadow_student_value_std_list[index_i])
                # mean_teacher_teacher_next = np.mean(shadow_student_value_std_list[index_i])
                # var_teacher_teacher_next = np.var(shadow_student_value_std_list[index_i])
                sum_teacher_teacher_next_record.append(sum_teacher_teacher_next)
                # mean_teacher_teacher_next_record.append(mean_teacher_teacher_next)
                # var_teacher_teacher_next_record.append(var_teacher_teacher_next)         
                
                # sum_max_min = sum(shadow_student_value_max_min_list[index_i])
                # mean_max_min = np.mean(shadow_student_value_max_min_list[index_i])
                # var_max_min = np.var(shadow_student_value_max_min_list[index_i])
                # sum_max_min_record.append(sum_max_min)
                # mean_max_min_record.append(mean_max_min)
                # var_max_min_record.append(var_max_min)     
                
                
                # sum_max_mean_distance = sum(shadow_student_value_max_mean_distance_list[index_i])
                # mean_max_mean_distance = np.mean(shadow_student_value_max_mean_distance_list[index_i])
                # var_max_mean_distance = np.var(shadow_student_value_max_mean_distance_list[index_i])
                # sum_max_mean_distance_record.append(sum_max_mean_distance)
                # mean_max_mean_distance_record.append(mean_max_mean_distance)
                # var_max_mean_distance_record.append(var_max_mean_distance)     
                
                
                ## hypothesis test
                shadow_student_value_mean_vertical_record.append(shadow_student_value_mean_vertical_list[index_i])
                shadow_student_value_std_vertical_record.append(shadow_student_value_std_vertical_list[index_i])        
                suspected_model_value_mean_reward.append(np.mean(student1_data))
                suspected_model_value_std_reward.append(np.std(student1_data))
                
                shadow_value_deviation_abs_sum_l1_record.append(shadow_value_deviation_abs_sum_l1_list[index_i])
                
                
                teacher_student_l2_distance = np.sum(np.square(shadow_student_value_mean_list[index_i].squeeze() - student1_data.squeeze()))
                teacher_student_l2_distance_record.append(teacher_student_l2_distance)
                shadow_value_deviation_abs_sum_l2_record.append(shadow_value_deviation_abs_sum_l2_list[index_i])
                
                
                teacher_student_cos_distance_record.append(cosine(shadow_student_value_mean_list[index_i].squeeze(), student1_data.squeeze()))
                shadow_model_cos_distance_record.append(shadow_model_cos_distance_list[index_i])
                
                # cosine_value_list.append(cosine(shadow_student_value_stack[:, i], shadow_student_value_mean))
                # normal_weighted_cosine_value_list.append(cosine(shadow_student_value_stack[:, i], shadow_student_value_mean, np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze())))
                # wasserstein_distance_list.append(wasserstein_distance(shadow_student_value_stack[:, i], shadow_student_value_mean))
                # our_weighted_cosine_value_list.append(np.sum(np.abs(shadow_student_value_stack[:, i].squeeze() - shadow_student_value_mean.squeeze()) * shadow_student_value_stack[:, i].squeeze() * shadow_student_value_mean.squeeze())) 
                
                
                ## weighted cosine distance
                teacher_student_cos_distance_weighted_record.append(cosine(student1_data.squeeze(), shadow_student_value_mean_list[index_i].squeeze(), np.abs(student1_data.squeeze() - shadow_student_value_mean_list[index_i].squeeze())))
                shadow_model_cos_distance_weighted_record.append(shadow_model_cos_distance_weighted_list[index_i])
                

                ## wasserstein distance
                teacher_student_wasserstein_distance_record.append(wasserstein_distance(student1_data.squeeze(), shadow_student_value_mean_list[index_i].squeeze()))
                shadow_model_wasserstein_distance_record.append(shadow_model_wasserstein_distance_list[index_i])
                
                ## our weighted cosine
                teacher_student_our_weighted_cosine_record.append(np.sum(np.abs(student1_data.squeeze() - shadow_student_value_mean_list[index_i].squeeze()) * student1_data.squeeze() * shadow_student_value_mean_list[index_i].squeeze()))
                shadow_model_our_weighted_cosine_record.append(shadow_model_our_weighted_cosine_list[index_i])
                
                episode_record.append(index_i)
                actual_buffer_name_record.append(teacher_name)
                student_name_record.append(student_name)
                audit_buffer_name_record.append(audit_dataset_name)
                
        data_dict = {'episode': episode_record, 
                    'actual_buffer_name': actual_buffer_name_record,
                    'student_name': student_name_record, 
                    'audit_buffer_name': audit_buffer_name_record, 
                    
                    'teacher_student_l1_distance': teacher_student_l1_distance_record,
                    'shadow_model_l1_distance': shadow_value_deviation_abs_sum_l1_record,
                    
                    'teacher_student_sum_std': sum_teacher_teacher_next_record, 
        
                    'teacher_student_l2_distance': teacher_student_l2_distance_record,
                    'shadow_model_l2_distance': shadow_value_deviation_abs_sum_l2_record, 
                    
                    'teacher_student_cos_distance': teacher_student_cos_distance_record,
                    'shadow_model_cos_distance': shadow_model_cos_distance_record, 
                    
                    'teacher_student_scipy_cos_distance_weighted': teacher_student_cos_distance_weighted_record,
                    'shadow_model_scipy_cos_distance_weighted': shadow_model_cos_distance_weighted_record,
                    
                    'teacher_student_wasserstein_distance': teacher_student_wasserstein_distance_record,
                    'shadow_model_wasserstein_distance': shadow_model_wasserstein_distance_record, 
                    
                    'teacher_student_cos_distance_weighted': teacher_student_our_weighted_cosine_record, 
                    'shadow_model_cos_distance_weighted': shadow_model_our_weighted_cosine_record
                    
                    # 'mean_teacher_student1': mean_teacher_student1_record, 
                    # 'mean_obs_act_value_std': mean_teacher_teacher_next_record, 
                    
                    # 'var_teacher_student1': var_teacher_student1_record, 
                    # 'var_obs_act_value_std': var_teacher_teacher_next_record,
                    
                    # 'sum_max_min': sum_max_min_record,
                    # 'mean_max_min': mean_max_min_record,
                    # 'var_max_min': var_max_min_record, 
                    
                    # 'sum_max_mean_distance': sum_max_mean_distance_record,
                    # 'mean_max_mean_distance': mean_max_mean_distance_record,
                    # 'var_max_mean_distance': var_max_mean_distance_record, 
                    
                    # 'shadow_student_value_mean': shadow_student_value_mean_vertical_record,
                    # 'shadow_student_value_std': shadow_student_value_std_vertical_record,
                    # 'suspected_model_value_mean': suspected_model_value_mean_reward,
                    # 'suspected_model_value_std': suspected_model_value_std_reward, 
                    }
        
        # import pdb; pdb.set_trace()            
        df = pd.DataFrame.from_dict(data_dict)
        
        df.to_excel(os.path.join(result_save_path, 'audit_result-numepi_{}-audname_{}-critag_{}-sustype_{}-sustag_{}-numstu_{}-assemble_{}-{}.xlsx'.format(num_of_audited_episode, audit_dataset_name, critic_model_tag[:critic_model_tag.find(".")], suspected_model_type, suspected_model_tag, num_shadow_student, assemble_expt_hyp, get_time_stamp())))
        
        # df.to_csv(os.path.join(result_save_path, 'audit_result-num_of_episode_{}-audit_buffer_name_{}-critic_model_tag_{}-suspected_model_type_{}-suspected_model_tag_{}-num_shadow_student_{}-noise_std_{}-{}.csv'.format(num_of_audited_episode, audit_dataset_name, critic_model_tag[:critic_model_tag.find(".")], suspected_model_type, suspected_model_tag, num_shadow_student, noise_std, get_time_stamp())))


    @staticmethod
    def hypothesis_testing(result_save_path, anomaly_detection_method='audit_acc_grubbs', sigma=0.01):
        def audit_acc_sigma_range(data, sigma, used_metric):
            
            def is_gaussian_test(list):
                statistic, critical_values, significance_levellist = anderson(list)    
                return statistic < critical_values[2]
            
            def max_deviation(list, mean_value):
                return np.max(np.abs(list-mean_value))

            def string2list(str):
                return literal_eval(re.sub("(?<=[0-9]|\.)\s+", ",", str.strip()))
                

            # add use_buffer_flag
            data['use_buffer_flag'] = (data['actual_buffer_name'] == data['audit_buffer_name'])
            # data.head()

            data["shadow_student_value_mean_is_gaussian"] = False
            data["shadow_student_value_mean_mean"] = 0.0
            data["shadow_student_value_mean_std"] = 0.0
            data["shadow_student_value_max_deviation"] = 0.0
            data = data.reset_index()
            num_rows = len(data)
            # teacher_student_cos_distance,shadow_model_cos_distance
            # teacher_student_cos_distance_weighted, shadow_model_cos_distance_weighted

            shadow_model_used_metric = f"shadow_model_{used_metric}"
            suspected_model_used_metric = f"teacher_student_{used_metric}"

            for index_r in range(num_rows):
                data.at[index_r, "shadow_student_value_mean_is_gaussian"] = is_gaussian_test(string2list(data[shadow_model_used_metric][index_r]))
                data.at[index_r, "shadow_student_value_mean_mean"] = np.mean(string2list(data[shadow_model_used_metric][index_r]))
                data.at[index_r, "shadow_student_value_max_deviation"] = max_deviation(string2list(data[shadow_model_used_metric][index_r]), data["shadow_student_value_mean_mean"][index_r])
                data.at[index_r, "shadow_student_value_mean_std"] = np.std(string2list(data[shadow_model_used_metric][index_r]))

            
            data["sum_teacher_student1"] = abs(data[suspected_model_used_metric] - data["shadow_student_value_mean_mean"])

            ratio_satisfying_gaussian = data["shadow_student_value_mean_is_gaussian"].sum()/num_rows * 100

            data['within_sigma'] = (data["sum_teacher_student1"] <= (sigma * data["shadow_student_value_mean_std"]))
            data['within_max-min-range'] = True
            data['audit_result'] = data['within_sigma'] & data['within_max-min-range']

            if len(data['use_buffer_flag']) > 0: 
                acc_number = sum(data['use_buffer_flag'] == data['audit_result'])
                overall_acc = sum(data['use_buffer_flag'] == data['audit_result'])/(len(data['use_buffer_flag']))
            else: overall_acc = -1

            tp_data = data[data['use_buffer_flag'] == True]
            if len(tp_data['use_buffer_flag']) > 0:
                true_positive_acc = sum(tp_data['use_buffer_flag'] == tp_data['audit_result'])/(len(tp_data['use_buffer_flag']))
            else: true_positive_acc = -1

            tn_data = data[data['use_buffer_flag'] == False]
            if len(tn_data['use_buffer_flag']) > 0:
                true_negative_acc = sum(tn_data['use_buffer_flag'] == tn_data['audit_result'])/(len(tn_data['use_buffer_flag']))
            else: true_negative_acc = -1

            acc_rate = f'{overall_acc*100:.2f}'
            tp_result = f'{true_positive_acc*100:.2f}'
            tn_result = f'{true_negative_acc*100:.2f}'

            return acc_rate, acc_number, tp_result, tn_result


        def audit_acc_grubbs(data, sigma, used_metric):
            
            def is_gaussian_test(list):
                statistic, critical_values, significance_levellist = anderson(list)    
                return statistic < critical_values[2]
            
            def max_deviation(list, mean_value):
                return np.max(np.abs(list-mean_value))

            def string2list(str):
                return literal_eval(re.sub("(?<=[0-9]|\.)\s+", ",", str.strip()))

            def grubbs(test_data, sigma):

                stdDev = np.std(test_data)
                mean = np.mean(test_data)
                tail_average = test_data[-1]
                z_score = (tail_average - mean) / stdDev
                len_series = len(test_data)
                threshold = scipy.stats.t.isf(sigma / (len_series), len_series - 2)  # one-side test
                # threshold = scipy.stats.t.isf(sigma / (2 * len_series), len_series - 2) # two-side test
                threshold_squared = threshold * threshold
                grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))
            
                return z_score <= grubbs_score
            
            # add use_buffer_flag
            data['use_buffer_flag'] = (data['actual_buffer_name'] == data['audit_buffer_name'])
            # data.head()

            data["shadow_student_value_mean_is_gaussian"] = False
            data["shadow_student_value_mean_mean"] = 0.0
            data["shadow_student_value_mean_std"] = 0.0
            data["shadow_student_value_max_deviation"] = 0.0
            data = data.reset_index()
            num_rows = len(data)
            # teacher_student_cos_distance,shadow_model_cos_distance
            # teacher_student_cos_distance_weighted, shadow_model_cos_distance_weighted

            shadow_model_used_metric = f"shadow_model_{used_metric}"
            suspected_model_used_metric = f"teacher_student_{used_metric}"
            
            for index_r in range(num_rows):
                data.at[index_r, "shadow_student_value_mean_is_gaussian"] = is_gaussian_test(string2list(data[shadow_model_used_metric][index_r]))
            
            ratio_satisfying_gaussian = data["shadow_student_value_mean_is_gaussian"].sum()/num_rows * 100

            data['within_sigma'] = False

                
            for index_r in range(num_rows):
                audit_data = string2list(data[shadow_model_used_metric][index_r])
                audit_data.append(data[suspected_model_used_metric][index_r])
                data.at[index_r, 'within_sigma'] = grubbs(audit_data, sigma)


            data['within_max-min-range'] = True
            data['audit_result'] = data['within_sigma'] & data['within_max-min-range']

            if len(data['use_buffer_flag']) > 0: 
                acc_number = sum(data['use_buffer_flag'] == data['audit_result'])
                overall_acc = sum(data['use_buffer_flag'] == data['audit_result'])/(len(data['use_buffer_flag']))
            else: overall_acc = -1

            tp_data = data[data['use_buffer_flag'] == True]
            if len(tp_data['use_buffer_flag']) > 0:
                true_positive_acc = sum(tp_data['use_buffer_flag'] == tp_data['audit_result'])/(len(tp_data['use_buffer_flag']))
            else: true_positive_acc = -1

            tn_data = data[data['use_buffer_flag'] == False]
            if len(tn_data['use_buffer_flag']) > 0:
                true_negative_acc = sum(tn_data['use_buffer_flag'] == tn_data['audit_result'])/(len(tn_data['use_buffer_flag']))
            else: true_negative_acc = -1

            acc_rate = f'{overall_acc*100:.2f}'
            tp_result = f'{true_positive_acc*100:.2f}'
            tn_result = f'{true_negative_acc*100:.2f}'

            return acc_rate, acc_number, tp_result, tn_result



        anomaly_detection_method = 'audit_acc_grubbs'
        used_metric_list = ['l1_distance', 'l2_distance', 'cos_distance', 'wasserstein_distance']

        if 'audit_acc_sigma_range' == anomaly_detection_method:
            all_results = {'anomaly detection': 'audit_acc_sigma_range', 'sigma': sigma, 'results': {}}

        elif 'audit_acc_grubbs' == anomaly_detection_method:
            all_results = {'anomaly detection': 'audit_acc_grubbs', 'sigma': sigma, 'results_between_each_two_datasets': {}, 'results_for_auditing_dataset':{}}
  
        
        for used_metric_index, used_metric in enumerate(used_metric_list):
            df_load = pd.read_excel(result_save_path)
            df_load["actual_buffer_name"] = df_load["actual_buffer_name"].str.replace(".h5", "")
            df_load["actual_buffer_name"] = df_load["actual_buffer_name"].str.replace(".hdf5", "")

            df_load["audit_buffer_name"] = df_load["audit_buffer_name"].str.replace(".h5", "")
            df_load["audit_buffer_name"] = df_load["audit_buffer_name"].str.replace(".hdf5", "")

            actual_buffer_name_list = sorted(df_load['actual_buffer_name'].unique().tolist())
            print(actual_buffer_name_list)

            audit_buffer_name_list = sorted(df_load['audit_buffer_name'].unique().tolist())
            print(audit_buffer_name_list)
            
            temp_true_positive_models = 0
            temp_all_positive_models = 0
            
            temp_true_negative_models = 0
            temp_all_negative_models = 0
            
            for _, audit_buffer_name in enumerate(audit_buffer_name_list):
                for _, actual_buffer_name in enumerate(actual_buffer_name_list):
                    df_load_copy = copy.deepcopy(df_load)
                    df_load_copy['audit_selected'] = (df_load_copy['actual_buffer_name']==actual_buffer_name) & (df_load_copy['audit_buffer_name']==audit_buffer_name)
                    df_load_copy = df_load_copy[df_load_copy['audit_selected']==True]
                    
                    if 'audit_acc_sigma_range' == anomaly_detection_method:
                        acc_result, acc_number, _, _ = audit_acc_sigma_range(df_load_copy, sigma, used_metric)
                    elif 'audit_acc_grubbs' == anomaly_detection_method:
                        acc_result, acc_number, _, _  = audit_acc_grubbs(df_load_copy, sigma, used_metric)                    

                    print(f"{actual_buffer_name}-{audit_buffer_name}-{acc_result}")
                    
                    if 0 == used_metric_index:
                        all_results['results_between_each_two_datasets']['actual_buffer_name: {}-audit_buffer_name: {}'.format(actual_buffer_name.split('_')[3], audit_buffer_name.split('_')[3])] = {}
                    
                    all_results['results_between_each_two_datasets']['actual_buffer_name: {}-audit_buffer_name: {}'.format(actual_buffer_name.split('_')[3], audit_buffer_name.split('_')[3])][f'{used_metric}'] = {'Accuracy': acc_result, 'Number of suspect model': df_load_copy.shape[0]}
                    
                    if actual_buffer_name == audit_buffer_name:
                        temp_true_positive_models += acc_number
                        temp_all_positive_models += df_load_copy.shape[0]
                    else:
                        temp_true_negative_models += acc_number
                        temp_all_negative_models += df_load_copy.shape[0]
                
                true_positive_rate = temp_true_positive_models/temp_all_positive_models
                true_negative_rate = temp_true_negative_models/temp_all_negative_models
                
                if 0 == used_metric_index:
                    all_results['results_for_auditing_dataset']['audit_buffer_name: {}'.format(audit_buffer_name.split('_')[3])] = {}
                
                all_results['results_for_auditing_dataset']['audit_buffer_name: {}'.format(audit_buffer_name.split('_')[3])][f'{used_metric}'] = {'True Positive Rate': true_positive_rate, 'True Negative Rate': true_negative_rate}
                    

        target_save_path = result_save_path.replace('xlsx', 'json')
        
        with open(target_save_path, 'w') as fp:
            json.dump(all_results, fp)

    

if __name__ == '__main__':
    print(os.getcwd())
    '''
    critic_type = 'DQN'
    teacher_buffer_name = '/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/teacher_buffer_in_transition_form/baseline/model_SAC_20221017142406518-500000.h5.npy'
    train_step = 500
    critic_save_path = '/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/trained_critic_model/baseline/model_SAC_20221017142406518-500000'
    num_of_cpu_processor = 10
    sorted_teacher_buffer_save_path = '/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/sorted_teacher_buffer_by_auditor/baseline'
    critic_model_tag = 'ckpt_500.pt'
    cuda_id = 0
    debug = True

    ## The setting is for 'LunarLanderContinuous-v2'
    # initial_dim_division = {'c0': np.zeros(2),
    #                         'c1': np.zeros(2),
    #                         'c2': np.zeros(2),
    #                         'c3': np.zeros(2),
    #                         'c4': np.zeros(2),
    #                         'c5': np.zeros(2),
    #                         'd6': [0, 1],
    #                         'd7': [0, 1],
    #                         'c6': np.zeros(2),
    #                         'c7': np.zeros(2)
    #                         } 
    # initial_dim_division_space = 4
    
    initial_dim_division = {'c0': np.zeros(2),
                            'c1': np.zeros(2)
                            } 
    initial_dim_division_space = 1

    # original_probe_output = np.load('/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_teacher_output/probe_input_by_auditor+baseline/model_SAC_20221017142514496/fingerprint_from_critic_model-length_of_observation_498747-20221023172053352-ckpt_500.pt.npy')
    original_probe_output = np.load('/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_teacher_output/probe_input_by_auditor+baseline/model_SAC_20221017142406518/fingerprint_from_critic_model-length_of_observation_498121-20221023172017802-ckpt_460.pt.npy')
    # suspected_model_output = np.load('/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_student_output_BC/probe_input_by_auditor+baseline/model_SAC_20221017142514496/fingerprint_from_critic_model-length_of_observation_498747-20221023172037338-ckpt_460.pt/model_SAC_20221017142514496-500000-BC_20221019233454_states_actions.npy')
    suspected_model_output = np.load('/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_student_output_BC/probe_input_by_auditor+baseline/model_SAC_20221017142406518/fingerprint_from_critic_model-length_of_observation_498121-20221023172017802-ckpt_460.pt/model_SAC_20221017142406518-500000-BC_20221019233454_states_actions.npy')

    parser = argparse.ArgumentParser("")
    parser.add_argument("--teacher_buffer_name", type=str, default='/home/c01lidu/CISPA-projects/rl_steal-2022/home/c01lidu/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole/auditor/teacher_buffer_in_transition_form/baseline/model_SAC_20221017142406518-500000.h5.npy')
    parser.add_argument("--critic_model_tag", type=str, default='ckpt_500.pt')
    args = parser.parse_args()


    if debug == True:
        my_auditor = MyAuditor(critic_type, 
                               teacher_buffer_name, 
                               train_step, 
                               critic_save_path, 
                               num_of_cpu_processor, 
                               sorted_teacher_buffer_save_path,
                               critic_model_tag,
                               cuda_id, 
                               debug)
        my_auditor.data_audit_v2(initial_dim_division, initial_dim_division_space, original_probe_output[:250, 8:], suspected_model_output[:250, 8:])
        # print(stats.ks_2samp(original_probe_output[:250, 8:].reshape(-1), suspected_model_output[:250, 8:].reshape(-1)))

    else:
        my_auditor = MyAuditor(critic_type, 
                               args.teacher_buffer_name, 
                               train_step, 
                               critic_save_path, 
                               num_of_cpu_processor, 
                               sorted_teacher_buffer_save_path,
                               args.critic_model_tag,
                               cuda_id, 
                               debug)


    '''


    '''
    ## print the estimated values of trained critic
    cuda_id = 0
    # device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    teacher_data_path_to_evaluate = '/Users/jinyang/Downloads/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_teacher_output/probe_input_by_auditor+baseline/model_SAC_20221017142406518/fingerprint_from_critic_model-length_of_observation_498121-20221023172017802-ckpt_460.pt.npy'
    student1_data_path_to_evaluate = '/Users/jinyang/Downloads/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_student_output_BC/probe_input_by_auditor+baseline/model_SAC_20221017142406518/fingerprint_from_critic_model-length_of_observation_498121-20221023172017802-ckpt_460.pt/model_SAC_20221017142406518-500000-BC_20221019233454_states_actions.npy'
    student2_data_path_to_evaluate = '/Users/jinyang/Downloads/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/probe_student_output_BC/probe_input_by_auditor+baseline/model_SAC_20221017142406518/fingerprint_from_critic_model-length_of_observation_498121-20221023172017802-ckpt_460.pt/model_SAC_20221017163911171-500000-BC_20221020003921_states_actions.npy'
    
    teacher_data = MyAuditor.value_estimation(teacher_data_path_to_evaluate, device)
    student1_data = MyAuditor.value_estimation(student1_data_path_to_evaluate, device)
    student2_data = MyAuditor.value_estimation(student2_data_path_to_evaluate, device)

    # print(teacher_data[:10]-student1_data[:10])
    # print(teacher_data[:10]-student2_data[:10])


    # evaluated_value_path = '/Users/jinyang/Downloads/off-rl/fingerprint_sample_generate_double-check/dqn_cartpole_copy/teacher_buffer/baseline/model_SAC_20221017142406518-500000.h5'
    # evaluated_value_mdpdataset = MDPDataset.load(evaluated_value_path)
    # for episode in evaluated_value_mdpdataset.episodes:
    #     print(episode)
    #     data_to_evaluate = np.concatenate((episode.observations, episode.actions), axis=1)
    #     # print(data_to_evaluate)
    #     MyAuditor.value_estimation(data_to_evaluate, device)
    '''
    

    '''
    ## 2022-11-26 Obsidian records
    ## For each episode, print the estimated values of trained critic
    cuda_id = 0
    gamma = 0.99
    env_name = 'LunarLanderContinuous-v2'
    device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
    student1_path = './fingerprint_sample_generate_double-check/dqn_cartpole_copy/student_policy/baseline/model_SAC_20221017142406518-500000/BC_20221019233454/model_500000.pt'
    student2_path = './fingerprint_sample_generate_double-check/dqn_cartpole_copy/student_policy/baseline/model_SAC_20221017163911171-500000/BC_20221019233454/model_500000.pt'
    
    # load student model
    drl_agent1 = BC()
    drl_agent1.build_with_env(gym.make(env_name))
    drl_agent1.load_model(student1_path)

    drl_agent2 = BC()
    drl_agent2.build_with_env(gym.make(env_name))
    drl_agent2.load_model(student2_path)
    

    evaluated_value_path = './fingerprint_sample_generate_double-check/dqn_cartpole_copy/teacher_buffer/baseline/model_SAC_20221017142406518-500000.h5'
    evaluated_value_mdpdataset = MDPDataset.load(evaluated_value_path)

    # data_record = []
    episode_record = []
    teacher_student_l1_distance_record = []
    sum_teacher_student2_record = []
    sum_teacher_teacher_next_record = []

    mean_teacher_student1_record = []
    mean_teacher_student2_record = []
    mean_teacher_teacher_next_record = []

    var_teacher_student1_record = []
    var_teacher_student2_record = []
    var_teacher_teacher_next_record = []

    rfi_student1_record = []
    rfi_student2_record = []

    for index_i, episode in enumerate(evaluated_value_mdpdataset.episodes):
        if index_i >= 50:kl kj kk k
            break
        print(f'episode: {index_i}')
        # print(episode)
        data_to_evaluate = np.concatenate((episode.observations, episode.actions), axis=1)
        
        # print(data_to_evaluate)
        teacher_data = MyAuditor.value_estimation(data_to_evaluate, device)
        teacher_data_next = episode.rewards[:-1]*0.01 + gamma * teacher_data[1:]

        probe_actions1 = drl_agent1.predict(episode.observations)
        probe_actions2 = drl_agent2.predict(episode.observations)

        data_to_evaluate1 = np.concatenate((episode.observations, probe_actions1), axis=1)
        data_to_evaluate2 = np.concatenate((episode.observations, probe_actions2), axis=1)

        
        student1_data = MyAuditor.value_estimation(data_to_evaluate1, device)
        student2_data = MyAuditor.value_estimation(data_to_evaluate2, device)

        # sum_teacher_student1 = sum(teacher_data - student1_data)
        # mean_teacher_student1 = np.mean(teacher_data - student1_data)
        # var_teacher_student1 = np.var(teacher_data - student1_data)
        
        # sum_teacher_student2 = sum(teacher_data - student2_data)
        # mean_teacher_student2 = np.mean(teacher_data - student2_data)
        # var_teacher_student2 = np.var(teacher_data - student2_data)


        sum_teacher_student1 = sum(abs(teacher_data[:-1] - student1_data[:-1]))
        mean_teacher_student1 = np.mean(teacher_data[:-1] - student1_data[:-1])
        var_teacher_student1 = np.var(teacher_data[:-1] - student1_data[:-1])
        
        sum_teacher_student2 = sum(abs(teacher_data[:-1] - student2_data[:-1]))
        mean_teacher_student2 = np.mean(teacher_data[:-1] - student2_data[:-1])
        var_teacher_student2 = np.var(teacher_data[:-1] - student2_data[:-1])
        
        sum_teacher_teacher_next = sum(abs(teacher_data[:-1] - teacher_data_next))
        mean_teacher_teacher_next = np.mean(teacher_data[:-1] - teacher_data_next)
        var_teacher_teacher_next = np.var(teacher_data[:-1] - teacher_data_next)
        


        counter1 = 0
        counter2 = 0

        for index_j in range(len(teacher_data)-1):
            if student1_data[index_j]>=teacher_data[index_j] and student1_data[index_j]<=teacher_data_next[index_j]:
                counter1+=1
            elif student1_data[index_j]<=teacher_data[index_j] and student1_data[index_j]>=teacher_data_next[index_j]:
                counter1+=1
            
            if student2_data[index_j]>=teacher_data[index_j] and student2_data[index_j]<=teacher_data_next[index_j]:
                counter2+=1
            elif student2_data[index_j]<=teacher_data[index_j] and student2_data[index_j]>=teacher_data_next[index_j]:
                counter2+=1
            
            print(f'{index_j} in episode')

        rfi_student1 = counter1/(len(teacher_data)-1)
        rfi_student2 = counter2/(len(teacher_data)-1)
        print(f'student 1: {rfi_student1}')
        print(f'student 2: {rfi_student2}')


        episode_record.append(index_i)
        teacher_student_l1_distance_record.append(sum_teacher_student1)
        sum_teacher_student2_record.append(sum_teacher_student2)
        sum_teacher_teacher_next_record.append(sum_teacher_teacher_next)

        mean_teacher_student1_record.append(mean_teacher_student1)
        mean_teacher_student2_record.append(mean_teacher_student2)
        mean_teacher_teacher_next_record.append(mean_teacher_teacher_next)

        var_teacher_student1_record.append(var_teacher_student1)
        var_teacher_student2_record.append(var_teacher_student2)
        var_teacher_teacher_next_record.append(var_teacher_teacher_next)
        
        rfi_student1_record.append(rfi_student1)
        rfi_student2_record.append(rfi_student2)

    data_dict = {'episode': episode_record, 
                'sum_teacher_student1': teacher_student_l1_distance_record, 
                'sum_teacher_student2': sum_teacher_student2_record, 
                'sum_teacher_teacher_next': sum_teacher_teacher_next_record, 
                
                'mean_teacher_student1': mean_teacher_student1_record, 
                'mean_teacher_student2': mean_teacher_student2_record, 
                'mean_teacher_teacher_next': mean_teacher_teacher_next_record, 
                
                'var_teacher_student1': var_teacher_student1_record, 
                'var_teacher_student2': var_teacher_student2_record, 
                'var_teacher_teacher_next': var_teacher_teacher_next_record, 

                'ratio_of_falling_in_student1': rfi_student1_record, 
                'ratio_of_falling_in_student2': rfi_student2_record}
        
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv('./test20221126.csv')
    '''


    
    ## Following the results of exp_20221126, we verify the teacher_teacher_next metric on other students. 
    ## 20221220 pack into function data_audit_v7
    cuda_id = 0
    gamma = 0.99
    env_name = 'LunarLanderContinuous-v2'
    device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')

    evaluated_value_path = 'fingerprint_sample_generate_double-check/dqn_cartpole_copy/teacher_buffer/baseline/model_SAC_20221017142406518-500000.h5'
    evaluated_value_mdpdataset = MDPDataset.load(evaluated_value_path)
    
    student_policy_path = 'fingerprint_sample_generate_double-check/dqn_cartpole_copy/student_policy/baseline'
    teacher_of_student_list = os.listdir(student_policy_path)
    
    # data_record
    episode_record = []
    actual_buffer_name_record = []
    audit_buffer_name_record = []
    teacher_student_l1_distance_record = []
    sum_teacher_student2_record = []
    sum_teacher_teacher_next_record = []

    mean_teacher_student1_record = []
    mean_teacher_student2_record = []
    mean_teacher_teacher_next_record = []

    var_teacher_student1_record = []
    var_teacher_student2_record = []
    var_teacher_teacher_next_record = []

    rfi_student1_record = []
    rfi_student2_record = []
    
    for teacher_name in teacher_of_student_list:
        student_path = os.path.join(student_policy_path, teacher_name, 'CQL_20221020222111', 'model_50000.pt')
    
        # load student model
        drl_agent = CQL()
        drl_agent.build_with_env(gym.make(env_name))
        drl_agent.load_model(student_path)
    
        for index_i, episode in enumerate(evaluated_value_mdpdataset.episodes):
            if index_i >= 50:
                break
            print(f'episode: {index_i}')
            # print(episode)
            data_to_evaluate = np.concatenate((episode.observations, episode.actions), axis=1)
            
            # print(data_to_evaluate)
            teacher_data = MyAuditor.value_estimation(data_to_evaluate, device)
            teacher_data_next = episode.rewards[:-1]*0.01 + gamma * teacher_data[1:]

            probe_actions1 = drl_agent.predict(episode.observations)
            data_to_evaluate1 = np.concatenate((episode.observations, probe_actions1), axis=1)
            student1_data = MyAuditor.value_estimation(data_to_evaluate1, device)

            
            sum_teacher_student1 = sum(abs(teacher_data[:-1] - student1_data[:-1]))
            mean_teacher_student1 = np.mean(abs(teacher_data[:-1] - student1_data[:-1]))
            var_teacher_student1 = np.var(abs(teacher_data[:-1] - student1_data[:-1]))
            
            sum_teacher_teacher_next = sum(abs(teacher_data[:-1] - teacher_data_next))
            mean_teacher_teacher_next = np.mean(abs(teacher_data[:-1] - teacher_data_next))
            var_teacher_teacher_next = np.var(abs(teacher_data[:-1] - teacher_data_next))
            
            
            
            # ## The average value of teacher_data[:-1] and teacher_data_next. 
            # teacher_data_avg = teacher_data[:-1] + teacher_data_next
            # sum_teacher_student1 = sum(abs(teacher_data_avg - student1_data[:-1]))
            # mean_teacher_student1 = np.mean(abs(teacher_data_avg - student1_data[:-1]))
            # var_teacher_student1 = np.var(abs(teacher_data_avg - student1_data[:-1]))
            
            # sum_teacher_teacher_next = sum(abs(teacher_data_avg - teacher_data_next))
            # mean_teacher_teacher_next = np.mean(abs(teacher_data_avg - teacher_data_next))
            # var_teacher_teacher_next = np.var(abs(teacher_data_avg - teacher_data_next))
            
            
            
            # sum_teacher_student1 = sum(abs(teacher_data_next - student1_data[:-1]))
            # mean_teacher_student1 = np.mean(abs(teacher_data_next - student1_data[:-1]))
            # var_teacher_student1 = np.var(abs(teacher_data_next - student1_data[:-1]))
            
            # sum_teacher_teacher_next = sum(abs(teacher_data[:-1] - teacher_data_next))
            # mean_teacher_teacher_next = np.mean(abs(teacher_data[:-1] - teacher_data_next))
            # var_teacher_teacher_next = np.var(abs(teacher_data[:-1] - teacher_data_next))
            
            
            counter1 = 0
            for index_j in range(len(teacher_data)-1):
                if student1_data[index_j]>=teacher_data[index_j] and student1_data[index_j]<=teacher_data_next[index_j]:
                    counter1+=1
                elif student1_data[index_j]<=teacher_data[index_j] and student1_data[index_j]>=teacher_data_next[index_j]:
                    counter1+=1
                print(f'{index_j} in episode')

            rfi_student1 = counter1/(len(teacher_data)-1)
            print(f'student 1: {rfi_student1}')

            episode_record.append(index_i)
            actual_buffer_name_record.append(teacher_name)
            audit_buffer_name_record.append('model_SAC_20221017142406518-500000')
            
            teacher_student_l1_distance_record.append(sum_teacher_student1)
            sum_teacher_teacher_next_record.append(sum_teacher_teacher_next)

            mean_teacher_student1_record.append(mean_teacher_student1)
            mean_teacher_teacher_next_record.append(mean_teacher_teacher_next)

            var_teacher_student1_record.append(var_teacher_student1)
            var_teacher_teacher_next_record.append(var_teacher_teacher_next)
            
            rfi_student1_record.append(rfi_student1)

    data_dict = {'episode': episode_record, 
                'teacher_name': actual_buffer_name_record,
                'evaluated_teacher_name': audit_buffer_name_record, 
                'sum_teacher_student1': teacher_student_l1_distance_record, 
                'sum_teacher_teacher_next': sum_teacher_teacher_next_record, 
                
                'mean_teacher_student1': mean_teacher_student1_record, 
                'mean_teacher_teacher_next': mean_teacher_teacher_next_record, 
                
                'var_teacher_student1': var_teacher_student1_record, 
                'var_teacher_teacher_next': var_teacher_teacher_next_record, 
                
                'ratio_of_falling_in_student1': rfi_student1_record}
        
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv('test20221129-CQL.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    