# from copyreg import pickle
import pickle
from dataclasses import replace
import os 
import copy
import gym
import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
import gym
from d3rlpy.algos import DiscreteBC, BC, NFQ, DQN, DoubleDQN, DiscreteSAC, BCQ, DiscreteBCQ, DiscreteCQL, CQL, TD3PlusBC, IQL

from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import evaluate_on_environment

from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split

class StudentTrain():
    def __init__(self, drl_type, buffer_save_path, drl_agent_save_path, cuda_id) -> None:
        self.drl_agent = drl_type
        self.buffer_save_path = buffer_save_path
        
        self.drl_agent_save_path = drl_agent_save_path
                               
        if 'lunarlander' in drl_agent_save_path:
            self.train_steps = 5_0000
        
        elif 'bipedalwalker' in drl_agent_save_path:
            self.train_steps = 10_0000            

        elif 'ant' in drl_agent_save_path:
            self.train_steps = 5_0000
        
        elif 'halfcheetah' in drl_agent_save_path:
            self.train_steps = 50_0000
                        
        else: print("Err: Env in student_train.py")

        self.cuda_id = cuda_id
        
        if drl_type == 'DiscreteBC':
            self.offline_DiscreteBC()
            
        elif drl_type == 'DiscreteBCQ':
            self.offline_DiscreteBCQ()
            
        elif drl_type == 'DiscreteSAC':
            self.offline_DiscreteSAC()

        elif drl_type == 'BC':
            self.offline_ContinuousBC()
            
        elif drl_type == 'BCQ':
            self.offline_ContinuousBCQ()

        elif drl_type == 'TD3PlusBC':
            self.offline_ContinuousTD3PlusBC()
            
        elif drl_type == 'IQL':
            self.offline_ContinuousIQL()
            
        else: print("Err: drl_type in student_train.py")
                     

    def offline_DiscreteBC(self):

        bc_agent = DiscreteBC(use_gpu=self.cuda_id)


        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        
        student_dir = self.buffer_save_path.split('/')[-1]


        log_dir = os.path.join(self.drl_agent_save_path, student_dir)

        bc_agent.fit(replay_dataset, 
                     n_steps=self.train_steps,
                     logdir=log_dir
                      )


    def offline_DiscreteBCQ(self):

        # setup algorithm
        # bcq_agent = DiscreteBCQ(learning_rate=1e-4, target_update_interval=100, batch_size = 512, use_gpu=self.cuda_id)
        # bcq_agent = DiscreteBCQ(action_flexibility=0.03, batch_size = 512, use_gpu=self.cuda_id)
        # bcq_agent = DiscreteBCQ(optim_factory=d3rlpy.models.optimizers.AdamFactory(amsgrad=True), batch_size=1024, use_gpu=self.cuda_id)
        # bcq_agent = DiscreteBCQ(learning_rate=3e-4, action_flexibility=5e-3, target_update_interval=100, use_gpu=self.cuda_id)
        # bcq_agent = DiscreteBCQ(target_update_interval=1000, batch_size=1024, use_gpu=self.cuda_id)
        # bcq_agent = DiscreteBCQ(action_flexibility=0.6, use_gpu=self.cuda_id)
        bcq_agent = DiscreteBCQ(use_gpu=self.cuda_id)
        
        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        

        student_dir = self.buffer_save_path.split('/')[-1]
        # student_dir = student_dir.replace('.h5', '')

        log_dir = os.path.join(self.drl_agent_save_path, student_dir)
        
        # def eval_in_training(algo, epoch, total_step, logdir):
        #     if total_step % 1000 == 0:
        #         env_name = 'CartPole-v1'
        #         env = gym.make(env_name)

        #         evaluate_scorer = evaluate_on_environment(n_trials=10, env=env)
        #         rewards_mean = evaluate_scorer(algo)

        #         if rewards_mean >= 500.0:
        #             model_path = os.path.join(logdir, f"model_{total_step}.pt")
        #             algo.save_model(model_path)
        #         print(f'{epoch, total_step}: {rewards_mean}')
            
        
        # env_name = 'CartPole-v1'
        # env = gym.make(env_name)
        bcq_agent.fit(replay_dataset, 
                      # eval_episodes=replay_dataset,
                      # n_epochs=10,
                      # n_steps_per_epoch = 1000,
                      n_steps=self.train_steps,
                      # shuffle=True, 
                    # tensorboard_dir=log_dir,
                    # callback=eval_in_training,
                    # scorers={'environment': d3rlpy.metrics.evaluate_on_environment(env)}
                    logdir=log_dir
                    )


    def offline_ContinuousBC(self):

        # setup algorithm
        bc_agent = BC(use_gpu=self.cuda_id)

        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        
        # env = gym.make('CartPole-v1')
        # evaluate_scorer = evaluate_on_environment(n_trials=10, env=env)
        # import pdb; pdb.set_trace()
        
        # train_episodes, test_episodes = train_test_split(replay_dataset.episodes)
        
        # transitions = []
        # for episode in replay_dataset.episodes:
        #     transitions.extend(episode.transitions)

        # log_dir = self.buffer_save_path.replace('teacher_buffer', 'student_policy')
        # log_dir = log_dir.replace('.h5', '')
        
        
        student_dir = self.buffer_save_path.split('/')[-1]
        # student_dir = student_dir.replace('.h5', '')

        log_dir = os.path.join(self.drl_agent_save_path, student_dir)

        bc_agent.fit(replay_dataset, 
                    # n_epochs=10,
                    n_steps=self.train_steps,
                    #shuffle=True, 
                    verbose=False,
                    logdir=log_dir, 
                    tensorboard_dir=log_dir
                    # eval_episodes=test_episodes, 
                    )


    def offline_ContinuousBCQ(self):

        # setup algorithm
        bcq_agent = BCQ(use_gpu=self.cuda_id)
        
        # prepare experience replay buffer
        # if isinstance(self.buffer_save_path, list):
        #     for buffer_index, buffer_path in enumerate(self.buffer_save_path):
        #         if buffer_index == 0:
        #            replay_dataset = MDPDataset.load(buffer_path)
        #         else:
        #            replay_dataset.extend(MDPDataset.load(buffer_path))
        # else:
        # replay_dataset = MDPDataset.load(self.buffer_save_path)

        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        
        student_dir = self.buffer_save_path.split('/')[-1]
        # student_dir = student_dir.replace('.h5', '')

        log_dir = os.path.join(self.drl_agent_save_path, student_dir)

        bcq_agent.fit(replay_dataset, 
                      # n_epochs=10,
                      # n_steps_per_epoch=1000, 
                      n_steps=self.train_steps,
                      verbose=False,
                      logdir=log_dir, 
                      tensorboard_dir=log_dir
                      # eval_episodes=test_episodes, 
                      )


    def offline_ContinuousTD3PlusBC(self):

        # setup algorithm
        td3bc_agent = TD3PlusBC(use_gpu=self.cuda_id)
        
        # prepare experience replay buffer
        # if isinstance(self.buffer_save_path, list):
        #     for buffer_index, buffer_path in enumerate(self.buffer_save_path):
        #         if buffer_index == 0:
        #            replay_dataset = MDPDataset.load(buffer_path)
        #         else:
        #            replay_dataset.extend(MDPDataset.load(buffer_path))
        # else:
        # replay_dataset = MDPDataset.load(self.buffer_save_path)

        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        
        student_dir = self.buffer_save_path.split('/')[-1]
        # student_dir = student_dir.replace('.h5', '')

        log_dir = os.path.join(self.drl_agent_save_path, student_dir)

        td3bc_agent.fit(replay_dataset, 
                      # n_epochs=10,
                      # n_steps_per_epoch=1000, 
                      n_steps=self.train_steps,
                      verbose=False,
                      logdir=log_dir, 
                      tensorboard_dir=log_dir
                      # eval_episodes=test_episodes, 
                      )
        
        
    def offline_ContinuousIQL(self):

        # setup algorithm
        iql_agent = IQL(use_gpu=self.cuda_id)
        
        # prepare experience replay buffer
        # if isinstance(self.buffer_save_path, list):
        #     for buffer_index, buffer_path in enumerate(self.buffer_save_path):
        #         if buffer_index == 0:
        #            replay_dataset = MDPDataset.load(buffer_path)
        #         else:
        #            replay_dataset.extend(MDPDataset.load(buffer_path))
        # else:
        # replay_dataset = MDPDataset.load(self.buffer_save_path)

        try:
            replay_dataset = MDPDataset.load(self.buffer_save_path)
        except OSError:
            with open(self.buffer_save_path, 'rb') as fp:
                # import pdb; pdb.set_trace()
                replay_dataset = pickle.load(fp)
            print('Replay_dataset\'s is Python List class')
        else:
            print('Replay_dataset\'s is MDPDataset class')
        
        student_dir = self.buffer_save_path.split('/')[-1]
        # student_dir = student_dir.replace('.h5', '')

        log_dir = os.path.join(self.drl_agent_save_path, student_dir)

        iql_agent.fit(replay_dataset, 
                      # n_epochs=10,
                      # n_steps_per_epoch=1000, 
                      n_steps=self.train_steps,
                      verbose=False,
                      logdir=log_dir, 
                      tensorboard_dir=log_dir
                      # eval_episodes=test_episodes, 
                      )
        
        
        
