from copy import deepcopy
# from turtle import pd

import os
import subprocess
import glob
import itertools
import pickle
import pathlib
import gym
import d3rlpy
import random
import numpy as np
from d3rlpy.algos import DiscreteBC, NFQ, DQN, DoubleDQN, DiscreteSAC, DiscreteBCQ, DiscreteCQL, BC, CQL, BCQ
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.dataset import MDPDataset


from lib_util.my_auditor import *

class BufferPreprocessor():
    def __init__(self, source_data_save_path, target_data_save_path) -> None:
        self.source_data_save_path = source_data_save_path
        self.target_data_save_path = target_data_save_path


    def add_gaussian_noise_to_buffer(self, env_name, noise_mu=0, noise_sigma=0):
        self.env = gym.make(env_name)
        self.observation_space_high = self.env.observation_space.high
        self.observation_space_low = self.env.observation_space.low
        # import pdb; pdb.set_trace()

        source_data_list = os.listdir(self.source_data_save_path)

        for source_data_name in source_data_list:
            replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, source_data_name))

            # import pdb; pdb.set_trace()
            gaussian_noise = np.random.normal(noise_mu, noise_sigma, size=replay_dataset.observations.shape)

            # replay_dataset.observations = replay_dataset.observations + gaussian_noise
            # AttributeError: can't set attribute

            perturbed_observations = replay_dataset.observations + gaussian_noise
            # clip to boundary of env
            clipped_perturbed_observations = np.clip(perturbed_observations, self.observation_space_low, self.observation_space_high)
            
            perturbed_dataset = d3rlpy.dataset.MDPDataset(observations=clipped_perturbed_observations,
                                                actions=replay_dataset.actions,
                                                rewards=replay_dataset.rewards,
                                                terminals=replay_dataset.terminals,
                                                episode_terminals=replay_dataset.episode_terminals
                                            )
            # import pdb; pdb.set_trace()



            # save MDPDataset
            save_dir_path = os.path.join(self.target_data_save_path, 'gaussian-mu_{}-sigma_{}'.format(noise_mu, noise_sigma))
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            perturbed_dataset.dump(os.path.join(save_dir_path, source_data_name))

    
    def mix_data_buffer_total_consistent(self, mixed_buffer_number):
        source_data_list = os.listdir(self.source_data_save_path)
        mixed_buffer_list = itertools.combinations(source_data_list, mixed_buffer_number)
        for mixed_buffer in mixed_buffer_list:
            sample_replay_dataset = []
            for buffer_index, buffer_data_name in enumerate(mixed_buffer):

                replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, buffer_data_name))
                sample_episode_total_number = int(len(replay_dataset)/mixed_buffer_number)
                sample_episode_index = np.random.choice(np.arange(len(replay_dataset)), size=sample_episode_total_number, replace=False)
                
                for _index in sample_episode_index:
                    sample_replay_dataset.append(deepcopy(replay_dataset[_index]))

                if buffer_index == 0:                    
                    mixed_buffer_save_name = buffer_data_name

                else:
                    mixed_buffer_save_name = mixed_buffer_save_name + '+' + buffer_data_name

            if len(sample_replay_dataset) < len(replay_dataset):
                padding_episode_number = len(replay_dataset) - len(sample_replay_dataset)
                sample_episode_index = np.random.choice(np.arange(len(replay_dataset)), size=padding_episode_number, replace=False)
                
                for _index in sample_episode_index:
                    sample_replay_dataset.append(deepcopy(replay_dataset[_index]))


            with open(os.path.join(self.target_data_save_path, mixed_buffer_save_name), "wb") as fp: 
                pickle.dump(sample_replay_dataset, fp)


    def mix_data_buffer_total_cumulated(self, mixed_buffer_number):
        source_data_list = os.listdir(self.source_data_save_path)
        mixed_buffer_list = itertools.combinations(source_data_list, mixed_buffer_number)
        for mixed_buffer in mixed_buffer_list:
            for buffer_index, buffer_data_name in enumerate(mixed_buffer):


                if buffer_index == 0:                    
                   replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, buffer_data_name))
                   mixed_buffer_save_name = buffer_data_name
                else:
                    replay_dataset.extend(MDPDataset.load(os.path.join(self.source_data_save_path, buffer_data_name)))
                    mixed_buffer_save_name = mixed_buffer_save_name + '+' + buffer_data_name

            # save MDPDataset
            replay_dataset.dump(os.path.join(self.target_data_save_path, mixed_buffer_save_name))


    def split_data_buffer(self, split_buffer_number):
        # split the dataset into disjoint subsets
        source_data_list = os.listdir(self.source_data_save_path)
        # mixed_buffer_list = itertools.combinations(source_data_list, mixed_buffer_number)
        for buffer_name in source_data_list:
            replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, buffer_name))
            
            random.shuffle(replay_dataset.episodes)
            replay_dataset_episode_num = len(replay_dataset)
            
            unit_replay_dataset_episode_num = int(replay_dataset_episode_num/split_buffer_number)
            
            for i in range(split_buffer_number):
                
                split_buffer_save_name = "{}-subdataset_{}.h5".format(buffer_name[:buffer_name.find(".")], i)
                obs = []
                act = []
                rew = []
                terminals = []
                                
                if (split_buffer_number-1) == i :
                    unit_replay_dataset = replay_dataset[i*unit_replay_dataset_episode_num:]
                    for _episode in unit_replay_dataset:
                        obs.append(deepcopy(_episode.observations))
                        act.append(deepcopy(_episode.actions))
                        rew.append(deepcopy(_episode.rewards))
                        
                        temp_terminals = np.zeros_like(_episode.rewards)
                        temp_terminals[-1] = 1
                        terminals.append(temp_terminals)
                    
                else:
                    unit_replay_dataset = replay_dataset[i*unit_replay_dataset_episode_num: (i+1)*unit_replay_dataset_episode_num]
                    for _episode in unit_replay_dataset:
                        obs.append(deepcopy(_episode.observations))
                        act.append(deepcopy(_episode.actions))
                        rew.append(deepcopy(_episode.rewards))
                        
                        temp_terminals = np.zeros_like(_episode.rewards)
                        temp_terminals[-1] = 1
                        terminals.append(temp_terminals)
                    
                obs_arr = np.concatenate(obs)
                act_arr = np.concatenate(act)
                rew_arr = np.concatenate(rew)
                terminals = np.concatenate(terminals)
                
                # save MDPDataset
                temp_dataset = MDPDataset(obs_arr, act_arr, rew_arr, terminals)          
                temp_dataset.dump(os.path.join(self.target_data_save_path, split_buffer_save_name))



    def split_data_buffer_overlaping(self, split_buffer_number, model_indice):

        source_data_list = os.listdir(self.source_data_save_path)
        # mixed_buffer_list = itertools.combinations(source_data_list, mixed_buffer_number)
        for buffer_name in source_data_list:
            replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, buffer_name))
            
            # random.shuffle(replay_dataset.episodes)
            # sorted_replay_dataset = sorted(replay_dataset.episodes)
            replay_dataset_episode_num = len(replay_dataset)
            
            unit_replay_dataset_episode_num = int(replay_dataset_episode_num/split_buffer_number)
            
            for i in range(split_buffer_number):
                
                split_buffer_save_name = "{}-overlap_indice_{}-subdataset_{}.h5".format(buffer_name[:buffer_name.find(".")], model_indice, i)
                obs = []
                act = []
                rew = []
                terminals = []
                                
                # if (split_buffer_number-1) == i :
                for k in range(split_buffer_number-model_indice):
                    start_index = (i+k)%split_buffer_number
                    if (start_index+1)*unit_replay_dataset_episode_num > replay_dataset_episode_num:
                        unit_replay_dataset = replay_dataset[start_index*unit_replay_dataset_episode_num:]
                    else:
                        unit_replay_dataset = replay_dataset[start_index*unit_replay_dataset_episode_num: (start_index+1)*unit_replay_dataset_episode_num]
                                
                    for _episode in unit_replay_dataset:
                        obs.append(deepcopy(_episode.observations))
                        act.append(deepcopy(_episode.actions))
                        rew.append(deepcopy(_episode.rewards))
                        
                        temp_terminals = np.zeros_like(_episode.rewards)
                        temp_terminals[-1] = 1
                        terminals.append(temp_terminals)
                    

                obs_arr = np.concatenate(obs)
                act_arr = np.concatenate(act)
                rew_arr = np.concatenate(rew)
                terminals = np.concatenate(terminals)
                
                # save MDPDataset
                temp_dataset = MDPDataset(obs_arr, act_arr, rew_arr, terminals)          
                temp_dataset.dump(os.path.join(self.target_data_save_path, split_buffer_save_name))
                

    def mix_data_buffer_student_policy_divided(self, divided_number):
        target_dir_list = os.listdir(self.target_data_save_path)
        target_dir_list = sorted(target_dir_list)

        assert divided_number == len(target_dir_list), "The divided_number does not equal to the number of the target dirs!!!"

        for buffer_index, target_dir in enumerate(target_dir_list):
 
            subprocess.call("cp  -r  {}/*  {}".format(self.source_data_save_path, os.path.join(self.target_data_save_path, target_dir)), shell=True)

            target_file_list = os.listdir(os.path.join(self.target_data_save_path, target_dir))
            for target_file in target_file_list: 
                buffer_name = target_file.split('+')[buffer_index]

                subprocess.call("mv  {}  {}".format(os.path.join(self.target_data_save_path, target_dir, target_file), os.path.join(self.target_data_save_path, target_dir, buffer_name)))


    def convert_MDPdataset_to_transitions(self, env_name):
        self.env = gym.make(env_name)
        self.observation_space_high = self.env.observation_space.high
        self.observation_space_low = self.env.observation_space.low
        # import pdb; pdb.set_trace()

        if pathlib.Path("self.source_data_save_path").is_dir():
            source_data_list = os.listdir(self.source_data_save_path)

            for source_data_name in source_data_list:
                replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, source_data_name))

                # import pdb; pdb.set_trace()
                print('replay_dataset loaded')
                # replay_dataset.observations = replay_dataset.observations + gaussian_noise
                # AttributeError: can't set attribute
                transitions = np.concatenate((replay_dataset.observations[:-1], 
                                            replay_dataset.actions[:-1] if len(replay_dataset.actions[:-1].shape)>1 else np.expand_dims(replay_dataset.actions[:-1], axis=1), 
                                            # next observation and action
                                            replay_dataset.observations[1:], 
                                            replay_dataset.actions[1:]  if len(replay_dataset.actions[1:].shape)>1 else np.expand_dims(replay_dataset.actions[1:], axis=1),
                                            # reward
                                            replay_dataset.rewards[:-1].reshape(-1,1)), axis=1)

                terminal_index = np.where(replay_dataset.terminals[:-1]==1.0)
                new_transitions = np.delete(transitions, terminal_index, axis=0)

                print(new_transitions.shape)
                # save new transitions
                save_dir_path = os.path.join(self.target_data_save_path)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                np.save(os.path.join(save_dir_path, source_data_name), new_transitions)
        else:
            
            replay_dataset = MDPDataset.load(self.source_data_save_path)

            # import pdb; pdb.set_trace()
            print('replay_dataset loaded')
            # replay_dataset.observations = replay_dataset.observations + gaussian_noise
            # AttributeError: can't set attribute
            transitions = np.concatenate((replay_dataset.observations[:-1], 
                                        replay_dataset.actions[:-1] if len(replay_dataset.actions[:-1].shape)>1 else np.expand_dims(replay_dataset.actions[:-1], axis=1), 
                                        # next observation and action
                                        replay_dataset.observations[1:], 
                                        replay_dataset.actions[1:]  if len(replay_dataset.actions[1:].shape)>1 else np.expand_dims(replay_dataset.actions[1:], axis=1),
                                        # reward
                                        replay_dataset.rewards[:-1].reshape(-1,1)), axis=1)

            terminal_index = np.where(replay_dataset.terminals[:-1]==1.0)
            new_transitions = np.delete(transitions, terminal_index, axis=0)

            print(new_transitions.shape)
            # save new transitions
            
            np.save(self.target_data_save_path, new_transitions)
            
            
    def convert_MDPdataset_to_MCTarget(self, env_name):
        def reward_to_Gt(reward, gamma=0.99):
            temp_reward = np.zeros_like(reward)
            temp_reward[-1] = reward[-1]
            reward_length = len(temp_reward)
            for i in reversed(range(1, reward_length, 1)):
                temp_reward[i-1] = reward[i-1] + gamma*temp_reward[i]
            return temp_reward
                
                
        self.env = gym.make(env_name)
        self.observation_space_high = self.env.observation_space.high
        self.observation_space_low = self.env.observation_space.low

        source_data_list = os.listdir(self.source_data_save_path)

        for source_data_name in source_data_list:
            replay_dataset = MDPDataset.load(os.path.join(self.source_data_save_path, source_data_name))
            
            for episode in replay_dataset:
                
                # episode.observations
                # episode.actions
                reward_Gt = reward_to_Gt(episode.rewards)        


    def construct_MDPdataset_from_suspected_model(self, env_name, student_agent_type, student_model_tag):
        eval_env = gym.make(env_name)
        suspected_model_set_name_list = os.listdir(self.source_data_save_path)

            
        for student_set_name in suspected_model_set_name_list:
            student_agent_name_list = os.listdir(os.path.join(self.source_data_save_path, student_set_name))
            
            for student_agent_name in student_agent_name_list:
                if student_agent_type in student_agent_name:

                    # Discrete agent
                    if student_agent_type == 'DiscreteBC':
                        drl_agent = DiscreteBC()
                    elif student_agent_type == 'DiscreteCQL':
                        drl_agent = DiscreteCQL()
                    elif student_agent_type == 'DiscreteBCQ':
                        drl_agent = DiscreteBCQ()

                    # Continuous agent
                    elif student_agent_type == 'BC':
                        drl_agent = BC()
                    elif student_agent_type == 'CQL':
                        drl_agent = CQL()
                    elif student_agent_type == 'BCQ':
                        drl_agent = BCQ()

                    drl_agent.build_with_env(eval_env)
                    drl_agent.load_model(os.path.join(self.source_data_save_path,
                                                        student_set_name,
                                                        student_agent_name,
                                                        student_model_tag))

                    ori_dataset_set_path = self.source_data_save_path.replace('teacher_policy', 'auditor/ori_dataset')
                    ori_dataset_name_list = os.listdir(ori_dataset_set_path)
                    for ori_dataset_name in ori_dataset_name_list:
                        ori_dataset_path = os.path.join(ori_dataset_set_path, ori_dataset_name)
                        ori_dataset = MDPDataset.load(ori_dataset_path)
                        actions = drl_agent.predict(ori_dataset.observations)
                        sus_dataset = MDPDataset(ori_dataset.observations, actions, ori_dataset.rewards, ori_dataset.terminals)        
                        
                        buffer_save_path = self.target_data_save_path.replace('teacher_policy', 'teacher_buffer')
                        buffer_save_name = str(student_set_name) + '+' + str(student_agent_name) + '+' + student_model_tag + '+' + 'audit_dataset' + '+' + str(ori_dataset_name)
                        sus_dataset.dump(os.path.join(buffer_save_path, buffer_save_name))


    def construct_competitor_classifier_dataset_from_suspected_model(self, student_agent_type, student_model_tag, divided_number, cuda_id):
        
        split_buffer_name_list = glob.glob("{}/*.h5".format(self.source_data_save_path))
        split_buffer_name_list = sorted(split_buffer_name_list)
        
        student_name_list = glob.glob("{}/*/{}_*/{}".format(self.source_data_save_path.replace("teacher_buffer", "student_policy"), student_agent_type, student_model_tag))
        student_name_list = sorted(student_name_list)
        
        assert len(split_buffer_name_list) == len(student_name_list)
        
        new_dataset = {"x": [], "y":[], "buffer_name":[], "student_model_name":[]}
        max_episode_len = -1
        for i, split_buffer_name in enumerate(split_buffer_name_list):
            mdp_dataset = MDPDataset.load(split_buffer_name)
            
            new_dataset['buffer_name'].append(split_buffer_name)

            for j in range(divided_number):
                start_model_index = i - (i % divided_number)
                student_agent_name = student_name_list[start_model_index]
                
                new_dataset['student_model_name'].append(student_agent_name)
                
                json_path = student_agent_name.replace(f"{student_model_tag}", "params.json")
                drl_shadow = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=student_agent_type, cuda_id=cuda_id)
                drl_shadow.load_model(student_agent_name)
                
                for one_episode in mdp_dataset.episodes:
                    model_actions = drl_shadow.predict(one_episode.observations)
                    
                    if model_actions.shape[0] >= max_episode_len:
                        max_episode_len = model_actions.shape[0]
                        
                    new_dataset['x'].append(deepcopy(np.concatenate((model_actions, one_episode.actions), axis=1)))
                    
                    if i % divided_number == j:
                        new_dataset['y'].append(1)
                    else:
                        new_dataset['y'].append(0)
                
            if (i+1) % divided_number == 0:
                # specify the desired length
                desired_length = max_episode_len
                for k in range(len(new_dataset['x'])):
                    # compute the number of elements to add
                    num_to_add = max(0, desired_length - new_dataset['x'][k].shape[0])
                    # use numpy.pad to add elements
                    new_dataset['x'][k] = np.pad(new_dataset['x'][k], ((0, num_to_add), (0, 0)), mode='edge')
                
                new_dataset['x'] = np.stack(new_dataset['x'], axis=0)
                new_dataset['y'] = np.stack(new_dataset['y'], axis=0)
                assert new_dataset['x'].shape[0] == new_dataset['y'].shape[0]

                new_dataset_name = split_buffer_name[:split_buffer_name.find("-subdataset")]
                new_dataset_save_path = new_dataset_name.replace("teacher_buffer", "auditor/competitor_classifier_dataset")
                
                with open('{}-{}.pkl'.format(new_dataset_save_path, student_agent_type), 'wb') as handle:
                    pickle.dump(new_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                new_dataset = {"x": [], "y":[], "buffer_name":[], "student_model_name":[]}
                max_episode_len = -1
                


class DrlAgentPreprocessor():
    def __init__(self, 
                 source_data_save_path, 
                 target_data_save_path, 
                 model_tag) -> None:
        self.source_data_save_path = source_data_save_path
        self.target_data_save_path = target_data_save_path
        self.model_tag = model_tag

    def d3rlpy_agent_to_torchscript_agent(self):
        d3rlpy_agent_dir_list = os.listdir(self.source_data_save_path)
        for d3rlpy_agent in d3rlpy_agent_dir_list:
            
            if 'DQN' in d3rlpy_agent:
                drl_agent = DQN.from_json(os.path.join(self.source_data_save_path, d3rlpy_agent, 'params.json'))
            
            drl_agent.load_model(os.path.join(self.source_data_save_path, 
                                              d3rlpy_agent, 
                                              self.model_tag))
        
            if not os.path.exists(os.path.join(self.target_data_save_path, d3rlpy_agent)):
                os.mkdir(os.path.join(self.target_data_save_path, d3rlpy_agent))
            
            drl_agent.save_policy(os.path.join(self.target_data_save_path, 
                                                d3rlpy_agent, 
                                                self.model_tag))                                 


    def torchscript_agent_to_torchnnmodule(self):
        d3rlpy_agent_dir_list = os.listdir(self.source_data_save_path)
        for d3rlpy_agent in d3rlpy_agent_dir_list:
            
            if 'DQN' in d3rlpy_agent:
                drl_agent = DQN.from_json(os.path.join(self.source_data_save_path, d3rlpy_agent, 'params.json'))
            
            drl_agent.load_model(os.path.join(self.source_data_save_path, 
                                              d3rlpy_agent, 
                                              self.model_tag))
        
            if not os.path.exists(os.path.join(self.target_data_save_path, d3rlpy_agent)):
                os.mkdir(os.path.join(self.target_data_save_path, d3rlpy_agent))
            
            drl_agent.save_policy(os.path.join(self.target_data_save_path, 
                                                d3rlpy_agent, 
                                                self.model_tag))                                 


    def select_qualified_agent(self, student_agent_type):
        student_path_list = glob.glob(f"{self.source_data_save_path}/*/{student_agent_type}_*")
        for student_path in student_path_list:
            student_ckpt_list = glob.glob(f"{student_path}/*.pt")
            if len(student_ckpt_list) < 2:
                print(student_path)
                subprocess.run(['mv', student_path, student_path.replace(student_path.split("/")[-1], "low_performance_agent")])  

    def rename_qualified_agent(self, student_agent_type):
        student_path_list = glob.glob(f"{self.source_data_save_path}/*/{student_agent_type}_*")
        for student_path in student_path_list:
            student_ckpt_list = glob.glob(f"{student_path}/*.pt")
            if len(student_ckpt_list) >= 2:
                student_ckpt_list = sorted(student_ckpt_list)
                first_ckpt = student_ckpt_list[-1]
                ckpt_tag = first_ckpt.split("/")[-1]
                print(student_path)
                subprocess.run(['mv', first_ckpt, first_ckpt.replace(ckpt_tag, self.model_tag)])  


if __name__ == '__main__':
    
    preprocess_method = 'split_data_buffer'
