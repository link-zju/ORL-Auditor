from asyncio.trsock import TransportSocket
import os 
import copy
from turtle import Turtle
import gym
import d3rlpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.multiprocessing as multiprocessing
from collections import OrderedDict
from d3rlpy.algos import DQN
from d3rlpy.dataset import MDPDataset
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from random import sample


class ProbeInputGenerator():
    def __init__(self, 
                 attack_method, 
                 length_of_single_fingerprint, 
                 total_number_of_fingerprints,
                 teacher_buffer_save_path,

                 shadow_policy_save_path, 
                 shadow_policy_model_tag, 
                 probe_input_save_path, 
                 adv_generate_probe_input_save_path, 
                 number_episode_optimization, 
                 distance_k, 
                
                 debug,
                 num_of_cpu_processor,
                 cuda_id
                 ) -> None:
        
        ## The parameters for random_sample
        self.attack_method = attack_method
        self.length_of_single_fingerprint = length_of_single_fingerprint
        self.total_number_of_fingerprints = total_number_of_fingerprints
        self.teacher_buffer_save_path = teacher_buffer_save_path
        
        
        ## The additional parameters for IPGuard
        self.shadow_policy_save_path = shadow_policy_save_path
        self.shadow_policy_model_tag = shadow_policy_model_tag
        self.probe_input_save_path = probe_input_save_path
        self.adv_generate_probe_input_save_path = adv_generate_probe_input_save_path
        self.number_episode_optimization = number_episode_optimization
        self.distance_k = distance_k
        
        
        ## The experiment parameters
        self.debug = debug
        self.num_of_cpu_processor = num_of_cpu_processor
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:{}".format(cuda_id))
        self.cuda_device = cuda_device
        
        
        if self.attack_method == 'fingerprint_from_random_sample':
            self.fingerprint_from_random_sample()
        elif self.attack_method == 'fingerprint_from_IPGuard':
            self.fingerprint_from_IPGuard()
        elif self.attack_method == 'fingerprint_from_all_buffer':
            print('fingerprint_from_all_buffer')
            self.fingerprint_from_all_buffer()
    
    def fingerprint_from_random_sample(self):

        self.teacher_buffer_list = os.listdir(self.teacher_buffer_save_path)
        
        for buffer_index, buffer_name in enumerate(self.teacher_buffer_list):
            teacher_buffer_path = os.path.join(self.teacher_buffer_save_path, buffer_name)
            replay_dataset = MDPDataset.load(teacher_buffer_path)
            
            for number_index in range(self.total_number_of_fingerprints):
                
                sample_index_list = np.random.randint(0, len(replay_dataset.observations), size=self.length_of_single_fingerprint)
                sampled_transitions = replay_dataset.observations[sample_index_list]
                # import pdb; pdb.set_trace()
                fingerprint_save_path = teacher_buffer_path.replace('teacher_buffer', 'probe_input')[:-3]
                fingerprint_file_name = 'fingerprint_from_random_sample' \
                                        + '-length_of_observation_{}-'.format(self.length_of_single_fingerprint) \
                                        + time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
                
                
                if not os.path.exists(fingerprint_save_path): 
                    os.makedirs(fingerprint_save_path)
                else:
                    pass
                
                np.save(os.path.join(fingerprint_save_path, fingerprint_file_name), sampled_transitions)
                time.sleep(1)


    @staticmethod
    def adv_fingerprint_generate(policy_neural_network, 
                                    probe_input, 
                                    number_episode_optimization, 
                                    distance_k,
                                    cuda_device, 
                                    adv_generate_probe_input_save_path,
                                    probe_input_dir,
                                    probe_input_name
                                    ):
        
        if type(probe_input).__module__ == np.__name__:
            probe_input_tensor = torch.from_numpy(probe_input).to(cuda_device)
        else:
            probe_input_tensor = probe_input.to(cuda_device)

        optimizer = torch.optim.Adam([probe_input_tensor.requires_grad_()], lr = 1e-3)

        for episode in range(number_episode_optimization):
            y_logits = policy_neural_network(probe_input_tensor)

            sorted, indices = torch.sort(y_logits)

            loss = F.relu(sorted[:, -1] - sorted[:, -2] + distance_k).sum()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss.item() == 0:
                break
        
        adv_probe_input = probe_input_tensor.cpu().detach().numpy()

        if not os.path.exists(os.path.join(adv_generate_probe_input_save_path, 'distance_k_{}'.format(distance_k), probe_input_dir)):
            os.makedirs(os.path.join(adv_generate_probe_input_save_path, 'distance_k_{}'.format(distance_k), probe_input_dir))

        np.save(os.path.join(adv_generate_probe_input_save_path, 'distance_k_{}'.format(distance_k), probe_input_dir, probe_input_name), adv_probe_input)
        
    def fingerprint_from_IPGuard(self):
        def torchscript_to_nnmodule(shadow_policy):
            policy_neural_network = nn.Sequential(OrderedDict([('l0', nn.Linear(4, 256)),
                                                                ('relu0', nn.ReLU()),
                                                                ('l1', nn.Linear(256, 256)),
                                                                ('relu1', nn.ReLU()),
                                                                ('l2', nn.Linear(256, 2)),
                                                                ('relu2', nn.ReLU())]))

            policy_neural_network.l0.weight.data = shadow_policy.code_with_constants[1].const_mapping['c0']
            policy_neural_network.l0.bias.data = shadow_policy.code_with_constants[1].const_mapping['c1']
            policy_neural_network.l1.weight.data = shadow_policy.code_with_constants[1].const_mapping['c2']
            policy_neural_network.l1.bias.data = shadow_policy.code_with_constants[1].const_mapping['c3']
            policy_neural_network.l2.weight.data = shadow_policy.code_with_constants[1].const_mapping['c4']
            policy_neural_network.l2.bias.data = shadow_policy.code_with_constants[1].const_mapping['c5']

            return policy_neural_network.to(self.cuda_device)


        if self.debug:
            self.probe_input_dir_list = os.listdir(self.probe_input_save_path)
            for probe_input_dir in self.probe_input_dir_list: 
                shadow_policy_dir = probe_input_dir.split('-')[0]
                shadow_policy = torch.jit.load(os.path.join(self.shadow_policy_save_path, shadow_policy_dir, self.shadow_policy_model_tag))
                policy_neural_network = torchscript_to_nnmodule(shadow_policy)

                self.probe_input_list = os.listdir(os.path.join(self.probe_input_save_path, probe_input_dir))
                for probe_input_name in self.probe_input_list:
                    probe_input = np.load(os.path.join(self.probe_input_save_path, probe_input_dir, probe_input_name))
                    self.adv_fingerprint_generate(policy_neural_network, probe_input, self.number_episode_optimization, self.distance_k, self.cuda_device, self.adv_generate_probe_input_save_path, probe_input_dir, probe_input_name)
  

        else:
            self.probe_input_dir_list = os.listdir(self.probe_input_save_path)
            
            torch.multiprocessing.set_start_method('spawn')
            with multiprocessing.Pool(processes=self.num_of_cpu_processor) as pool:

                for probe_input_dir in self.probe_input_dir_list: 
                    shadow_policy_dir = probe_input_dir.split('-')[0]
                    shadow_policy = torch.jit.load(os.path.join(self.shadow_policy_save_path, shadow_policy_dir, self.shadow_policy_model_tag))
                    policy_neural_network = torchscript_to_nnmodule(shadow_policy)

                    self.probe_input_list = os.listdir(os.path.join(self.probe_input_save_path, probe_input_dir))
                    for probe_input_name in self.probe_input_list:
                        probe_input = np.load(os.path.join(self.probe_input_save_path, probe_input_dir, probe_input_name))
                        
                        pool.apply_async(self.adv_fingerprint_generate, (policy_neural_network, probe_input, self.number_episode_optimization, self.distance_k, self.cuda_device, self.adv_generate_probe_input_save_path, probe_input_dir, probe_input_name, ))
                        print('probe_input_dir {}'.format(probe_input_dir))

                pool.close()
                pool.join()
                print('Sub-process done.')
   
    def fingerprint_from_all_buffer(self):
        self.teacher_buffer_list = os.listdir(self.teacher_buffer_save_path)
        
        for buffer_index, buffer_name in enumerate(self.teacher_buffer_list):
            teacher_buffer_path = os.path.join(self.teacher_buffer_save_path, buffer_name)
            replay_dataset = MDPDataset.load(teacher_buffer_path)
            
            
            sampled_transitions = replay_dataset.observations # all buffer of teacher model
            # import pdb; pdb.set_trace()
            fingerprint_save_path = teacher_buffer_path.replace('teacher_buffer', 'probe_input')[:-3]
            fingerprint_file_name = 'fingerprint_from_random_sample' \
                                    + '-length_of_observation_{}-'.format(sampled_transitions.shape[0]) \
                                    + time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
            
            
            if not os.path.exists(fingerprint_save_path): 
                os.makedirs(fingerprint_save_path)
            else:
                pass
            
            np.save(os.path.join(fingerprint_save_path, fingerprint_file_name), sampled_transitions)
            time.sleep(1)




        