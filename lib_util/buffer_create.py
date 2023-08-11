import os 
import copy
import gym
import d3rlpy
from d3rlpy.algos import DQN
import numpy as np
import stable_baselines3 as sb3
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.dataset import MDPDataset
from d3rlpy.wrappers.sb3 import to_mdp_dataset

def dataset_generation(model, env, buffer_save_path, buffer_length=50000):

    observations = []
    actions = []
    rewards = []
    terminals = []

    obs = env.reset()
    for i in range(buffer_length):
        observations.append(obs)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)

        # env.render()
        if done:
            obs = env.reset()
        print(i)

    env.close()
    
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)

    dataset = MDPDataset(observations, actions, rewards, terminals)
    dataset.dump(buffer_save_path + '-' + str(buffer_length) + '.h5')

class BufferCreate():
    def __init__(self, drl_agent_path, teacher_agent_from, env_name, buffer_length, buffer_save_path, create_method) -> None:            
        self.env_name = env_name
        self.drl_agent_path = drl_agent_path
        self.teacher_agent_from = teacher_agent_from
        self.buffer_save_path = buffer_save_path

        self.buffer_length = buffer_length
        
        self.env = gym.make(self.env_name)
        self.eval_env = gym.make(self.env_name)
        
        if create_method == 'trained':
            print(self.teacher_agent_from)
            if self.teacher_agent_from == 'd3rlpy':
                # setup algorithm
                self.drl_agent = d3rlpy.algos.DQN()

                # initialize neural networks before loading parameters
                self.drl_agent.build_with_env(self.env)

                # load pretrained parameters
                self.drl_agent.load_model(self.drl_agent_path)

                # prepare experience replay buffer
                buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=self.buffer_length, env=self.env)

                # start data collection
                self.drl_agent.collect(self.env, buffer, n_steps=self.buffer_length)

                # export as MDPDataset
                dataset = buffer.to_mdp_dataset()

                # save MDPDataset
                dataset.dump(self.buffer_save_path + '-' + str(self.buffer_length) + '.h5')
                
            if self.teacher_agent_from == 'stable-baselines3':
                # setup algorithm
                drl_agent_type = self.drl_agent_path.split("/")[-1].split("_")[1]  
                if "sac" == drl_agent_type:
                    self.drl_agent = sb3.SAC("MlpPolicy", env=self.env_name, device='cpu') 
                elif "ppo" == drl_agent_type:
                    self.drl_agent = sb3.PPO("MlpPolicy", env=self.env_name, device='cpu') 

                # initialize neural networks before loading parameters
                # self.drl_agent.build_with_env(self.env)

                # load pretrained parameters
                print('self.teacher_agent_from == stable-baselines3')
                self.drl_agent = self.drl_agent.load(self.drl_agent_path)
                print('self.drl_agent = self.drl_agent.load(self.drl_agent_path)')

                dataset_generation(self.drl_agent, self.env, self.buffer_save_path, self.buffer_length)
                
        
        elif create_method == 'random':
            # setup algorithm
            random_policy = d3rlpy.algos.DiscreteRandomPolicy()
            # prepare experience replay buffer
            buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=self.buffer_length, env=self.env)

            # start data collection
            random_policy.collect(self.env, buffer, n_steps=self.buffer_length)

            # export as MDPDataset
            dataset = buffer.to_mdp_dataset()

            # save MDPDataset
            dataset.dump(self.buffer_save_path + '-' + str(self.buffer_length) + '.h5')
