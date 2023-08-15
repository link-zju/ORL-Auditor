
import os
import re
from turtle import pd

import gym
import glob
from copy import deepcopy
import numpy as np
import pandas as pd
import random
import torch

import stable_baselines3 as sb3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from d3rlpy.metrics.scorer import evaluate_on_environment
import lib_util.parameter_setting as para
from lib_util.buffer_create import BufferCreate
from lib_util.student_train import StudentTrain
from lib_util.teacher_train import TeacherModelTrain
from lib_util.my_auditor import MyAuditor
from lib_util.file_tag import get_time_stamp
from lib_util.data_preprocessor import BufferPreprocessor

def print_error(value):
    print("error: ", value)


class MainExperiment():
    def __init__(self,
                 which_experiment,
                 env_name,
                 result_root_path,
                 teacher_save_path,
                 teacher_train_times,
                 teacher_reward_threshold,
                 teacher_model_tag,
                 teacher_buffer_save_path,
                 teacher_buffer_length,
                 teacher_buffer_create_method,
                 num_of_cpu_processor,
                 num_of_gpu,

                 student_agent_type,
                 student_save_path,
                 teacher_buffer_mix_flag,

                 fingerprint_generate_method,
                 length_of_single_fingerprint,
                 total_number_of_fingerprint,

                 shadow_policy_save_path,
                 shadow_policy_model_tag,
                 probe_input_save_path,
                 adv_generate_probe_input_save_path,
                 number_episode_optimization,
                 distance_k,

                 mixed_buffer_divided_number,
                 teacher_buffer_is_mixed_buffer,
                 student_model_tag,

                 teacher_agent_type,
                 multi_experiment_settings,
                 save_both_states_and_actions,
                 expt_name,
                 cuda_id,
                 expt_result_save_path,
                 teacher_agent_from,

                 num_of_audited_episode,
                 num_shadow_student, 
                 critic_model_tag,
                 probe_student_output_name,  # temporary variable
                 
                 trajectory_size, 
                 trajectory_splitting_num, 
                 
                 significance_level,

                 random_seed, 
                 debug=False,
                 ) -> None:

        self.result_root_path = result_root_path

        self.random_seed = random_seed

        self.env_name = env_name
        self.train_env = gym.make(env_name)
        self.train_env.seed(self.random_seed)

        self.eval_env = gym.make(env_name)
        self.eval_env.seed(self.random_seed)

        self.teacher_save_path = teacher_save_path
        self.teacher_train_times = teacher_train_times
        self.teacher_reward_threshold = teacher_reward_threshold

        # create teacher buffer
        self.teacher_model_tag = teacher_model_tag
        self.teacher_buffer_save_path = teacher_buffer_save_path
        self.teacher_buffer_length = teacher_buffer_length
        self.teacher_buffer_create_method = teacher_buffer_create_method
        self.num_of_cpu_processor = num_of_cpu_processor
        self.num_of_gpu = num_of_gpu

        # train student model
        self.student_agent_type = student_agent_type
        self.student_save_path = student_save_path
        self.teacher_buffer_mix_flag = teacher_buffer_mix_flag

        # collect probe input
        # self.teacher_buffer_save_path = teacher_buffer_save_path
        self.fingerprint_generate_method = fingerprint_generate_method
        self.length_of_single_fingerprint = length_of_single_fingerprint
        self.total_number_of_fingerprint = total_number_of_fingerprint

        ## The additional parameters for IPGuard
        self.shadow_policy_save_path = shadow_policy_save_path
        self.shadow_policy_model_tag = shadow_policy_model_tag
        self.probe_input_save_path = probe_input_save_path
        self.adv_generate_probe_input_save_path = adv_generate_probe_input_save_path
        self.number_episode_optimization = number_episode_optimization
        self.distance_k = distance_k

        # collect probe output
        self.mixed_buffer_divided_number = mixed_buffer_divided_number
        self.teacher_buffer_is_mixed_buffer = teacher_buffer_is_mixed_buffer
        self.student_model_tag = student_model_tag

        self.teacher_agent_type = teacher_agent_type

        self.multi_experiment_settings = multi_experiment_settings
        self.save_both_states_and_actions = True
        self.expt_name = expt_name
        self.cuda_id = cuda_id
        self.expt_result_save_path = expt_result_save_path
        self.teacher_agent_from = teacher_agent_from
        self.debug = debug

        self.num_of_audited_episode = num_of_audited_episode
        self.num_shadow_student = num_shadow_student
        self.critic_model_tag = critic_model_tag
        # self.probe_student_output_name = probe_student_output_name
        
        self.trajectory_size = trajectory_size
        self.trajectory_splitting_num = trajectory_splitting_num
        self.significance_level = significance_level
        
        
        if "train_teacher_model" == which_experiment:
            self.train_teacher_model()
            
        elif "eval_teacher_model" == which_experiment:
            self.eval_teacher_model()
            
        elif "teacher_buffer_create" == which_experiment:
            self.teacher_buffer_create()
            
        elif "train_student_model" == which_experiment:
            self.train_student_model()
        
        elif "eval_student_model" == which_experiment:
            self.eval_student_model()
        
        elif "eval_assemble_student_model" == which_experiment:
            self.eval_assemble_student_model()

        elif "auditor_train_critic_model" == which_experiment:
            self.auditor_train_critic_model()   ## train critic model     
        
        elif "audit_dataset" == which_experiment:
            self.audit_dataset()

        elif "state_action_value_record" == which_experiment:
            self.state_action_value_record()  ## record the state-action values for drawing t-SNE
        

    def train_teacher_model(self):
        TeacherModelTrain(self.env_name, 
                          self.teacher_save_path, 
                          self.teacher_train_times, 
                          self.random_seed, 
                          self.cuda_id)


    def eval_teacher_model(self):
        eval_env = gym.make(self.env_name)    

        eval_env = Monitor(eval_env)
        if ("sac" in self.teacher_save_path or "SAC" in self.teacher_save_path):
            teacher_model = sb3.SAC.load(self.teacher_save_path, device=self.cuda_id)
        elif ("ppo" in self.teacher_save_path or "PPO" in self.teacher_save_path):
            teacher_model = sb3.PPO.load(self.teacher_save_path, device=self.cuda_id)
            
        mean_reward, std_reward = evaluate_policy(teacher_model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"model_name={self.teacher_save_path.split('/')[-1]}")
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    def teacher_buffer_create(self):
        drl_agent_path = self.teacher_save_path
        
        if ".zip" in self.teacher_buffer_save_path:
            temp_index = self.teacher_buffer_save_path.rindex("/")
            self.teacher_buffer_save_path = self.teacher_buffer_save_path[:temp_index]

        if not os.path.exists(self.teacher_buffer_save_path):
            os.makedirs(self.teacher_buffer_save_path)

        if self.teacher_buffer_create_method == 'trained':
            buffer_save_name = drl_agent_path.split('/')[-1][:drl_agent_path.split('/')[-1].find('.')]
            buffer_save_path = os.path.join(self.teacher_buffer_save_path, buffer_save_name)

        elif self.teacher_buffer_create_method == 'random':
            buffer_save_path = os.path.join(self.teacher_buffer_save_path, 'random_policy')
            
        BufferCreate(drl_agent_path, 
                     self.teacher_agent_from, 
                     self.env_name, 
                     self.teacher_buffer_length,
                     buffer_save_path, 
                     self.teacher_buffer_create_method)


    def train_student_model(self):
        drl_type = self.student_agent_type
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        StudentTrain(drl_type, self.teacher_buffer_save_path, self.student_save_path, self.cuda_id)
            
            
    def eval_student_model(self):
        self.student_save_path = self.teacher_buffer_save_path.replace('teacher_buffer', 'student_policy')
        student_set_name = self.student_save_path.split('/')[-1]
        
        student_agent_path_list = glob.glob(f'{self.student_save_path}/{self.student_agent_type}_*/{self.student_model_tag}')
        record_results = []
        
        # import pdb;pdb.set_trace()
        
        for student_agent_path in student_agent_path_list:
            eval_result = {}
            student_agent_name = student_agent_path.split('/')[-2]
            print(f'student_set_name: {student_set_name}')
            print(f'student_agent_name: {student_agent_name}')
            
            json_path = student_agent_path.replace(f"{self.student_model_tag}", "params.json")
            drl_agent = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=self.student_agent_type, cuda_id=cuda_id)
            drl_agent.load_model(student_agent_path)

            env = gym.make(self.env_name)
            evaluate_scorer = evaluate_on_environment(n_trials=10, env=env)
            rewards_mean = evaluate_scorer(drl_agent)

            eval_result['teacher_buffer_name'] = student_set_name
            eval_result['student_model_name'] = student_agent_name
            eval_result['student_model_tag'] = self.student_model_tag
            eval_result['rewards_mean'] = rewards_mean

            record_results.append(eval_result)
            
        record_results_pd = pd.DataFrame.from_dict(record_results)
        record_results_pd.to_csv(os.path.join(self.expt_result_save_path, 'student_eval_by_env_results-{}-{}-{}-{}.csv'.format(self.env_name, self.student_agent_type, student_set_name, get_time_stamp())))
            
            
    def eval_assemble_student_model(self):
        def assemble_actions(assemble_drl_agent, observation):
            actions = []
            for single_drl_agent in assemble_drl_agent:
                actions.append(deepcopy(single_drl_agent.predict([observation])[0]))
            
            return np.mean(np.stack(actions, axis=1),axis=1)
            
        self.student_save_path = self.teacher_buffer_save_path.replace('teacher_buffer', 'student_policy')
        self.student_save_path = self.student_save_path.replace('baseline', 'overlap_split_5_indice_2')
        self.student_save_path = self.student_save_path[:self.student_save_path.find('.h')]
        
        student_agent_path_list = glob.glob(f'{self.student_save_path}*.h*5/{self.student_agent_type}_*/{self.student_model_tag}')
        
        # import pdb;pdb.set_trace()
        assert 5 == len(student_agent_path_list)
        
        indice_number = int(re.findall(r'indice_[0-9]+', self.student_save_path)[0].split("_")[-1]) 
        student_agent_path_list = random.sample(student_agent_path_list, indice_number)
        
        assert indice_number == len(student_agent_path_list)
        
        record_results = []
        
        assemble_drl_agent = []
        for student_agent_path in student_agent_path_list:
            eval_result = {}
            teacher_buffer_name = self.student_save_path.split('/')[-1]
            print(f'teacher_buffer_name: {teacher_buffer_name}')
            
            json_path = student_agent_path.replace(f"{self.student_model_tag}", "params.json")
            drl_agent = MyAuditor.load_agent_from_json(json_path=json_path, suspected_model_type=self.student_agent_type, cuda_id=cuda_id)
            drl_agent.load_model(student_agent_path)
            assemble_drl_agent.append(deepcopy(drl_agent))
        
        
        env = gym.make(self.env_name)
        episode_rewards = []
        n_trials = 10
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0
            
            while True:
                action = assemble_actions(assemble_drl_agent, observation)

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if done:
                    break
            episode_rewards.append(episode_reward)
        
        # return float(np.mean(episode_rewards))

            # evaluate_scorer = evaluate_on_environment(n_trials=10, env=env)
            # rewards_mean = evaluate_scorer(drl_agent)

        eval_result['teacher_buffer_name'] = teacher_buffer_name
        eval_result['student_agent_type'] = self.student_agent_type
        eval_result['student_model_tag'] = self.student_model_tag
        eval_result['rewards_mean'] = float(np.mean(episode_rewards))
        eval_result['rewards_std'] = float(np.std(episode_rewards))

        record_results.append(eval_result)
        record_results_pd = pd.DataFrame.from_dict(record_results)
        record_results_pd.to_csv(os.path.join(self.expt_result_save_path, 'assemble_student_eval_by_env_results-{}-{}-{}-{}-{}.csv'.format(self.env_name, teacher_buffer_name, self.student_agent_type, self.student_model_tag, get_time_stamp())))
           

    def auditor_train_critic_model(self):
        MDPdataset_save_path = self.teacher_buffer_save_path
        device = torch.device('cuda:{}'.format(cuda_id) if torch.cuda.is_available() else 'cpu')
        # /datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000_500000.h5
        
        # /datasets_and_models_set1/sac_lunarlander/auditor/teacher_buffer_in_transition_form/td_error/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000_500000.h5.npy
        
        transitions_save_path = MDPdataset_save_path.replace('teacher_buffer/baseline', 'auditor/teacher_buffer_in_transition_form/baseline') + '.npy'
        
        if not os.path.exists(transitions_save_path):        
            data_processor = BufferPreprocessor(MDPdataset_save_path, transitions_save_path)
            data_processor.convert_MDPdataset_to_transitions(self.env_name)
            
        
        MyAuditor.critic_generate_all_dataset(device, transitions_save_path)


    def audit_dataset(self):
        MyAuditor.data_audit(self.teacher_buffer_save_path,
                             self.critic_model_tag,
                             self.student_save_path,
                             self.expt_result_save_path,
                             self.student_model_tag,
                             self.student_agent_type,
                             self.num_of_audited_episode,

                             self.num_shadow_student, 
                             self.trajectory_size,
                             self.significance_level, 

                             self.env_name,
                             self.cuda_id)
        
        ## use for auditing assemble model in Section Robustness
        # MyAuditor.data_audit_assemble_model(self.teacher_buffer_save_path,
        #                                     self.critic_model_tag,
        #                                     self.student_save_path,
        #                                     self.expt_result_save_path,
        #                                     self.student_model_tag,
        #                                     self.student_agent_type,
        #                                     self.num_of_audited_episode,
        #                                     self.num_shadow_student, 
        #                                     self.env_name,
        #                                     self.cuda_id)
        
        ## use for the behavior similarity (action distance)
        # MyAuditor.action_distance(self.teacher_buffer_save_path,
        #                           self.critic_model_tag,
        #                           self.student_save_path,
        #                           self.expt_result_save_path,
        #                           self.student_model_tag,
        #                           self.student_agent_type,
        #                           self.num_of_audited_episode,
        #                           self.num_shadow_student, 
        #                           self.env_name,
        #                           self.cuda_id)

        
        ## record state-action values to draw TSNE
        # MyAuditor.state_action_value_record(self.teacher_buffer_save_path,
        #                                     self.critic_model_tag,
        #                                     self.student_save_path,
        #                                     self.expt_result_save_path,
        #                                     self.student_model_tag,
        #                                     self.student_agent_type,
        #                                     self.num_of_audited_episode,
        #                                     self.num_shadow_student, 
        #                                     self.env_name,
        #                                     self.cuda_id)
        


if __name__ == '__main__':
    
    load_use_args = True
    expt_setting = "baseline"
    
    
    if load_use_args:
        args = para.offlineRL_main_exp()
        
        datasets_and_models_set = args.datasets_and_models_dir

        env_name = args.env_name

        if 'LunarLanderContinuous-v2' == env_name:
            expt_dataset_name = 'sac_lunarlander'

        elif 'BipedalWalker-v3' == env_name:
            expt_dataset_name = 'sac_bipedalwalker'

        elif 'Ant-v2' == env_name:
            expt_dataset_name = 'sac_ant'
            
        elif 'HalfCheetah-v2' == env_name:
            expt_dataset_name = 'rluply_d4rl_halfcheetah'

        else: print('Error: expt_dataset_name illegal')

        which_experiment = args.which_experiment
        if "train_teacher_model" == args.which_experiment or "eval_teacher_model" == args.which_experiment or "teacher_buffer_create" == args.which_experiment:
            teacher_save_path = args.teacher_save_path
            expt_name = re.search(r'{}.+?/'.format(datasets_and_models_set), teacher_save_path)[0][:-1]

            project_data_root_path = re.search(f'.+(?=/{expt_name})', teacher_save_path)[0]
            
            result_root_path = os.path.join(project_data_root_path, expt_name)
            
            expt_result_save_path = os.path.join(project_data_root_path, 'result_save', expt_name)
            temp_name = os.path.join(result_root_path, expt_dataset_name)
            student_save_path = os.path.join(temp_name, 'student_policy/{}'.format(expt_setting))
            shadow_policy_save_path = os.path.join(temp_name, 'teacher_shadow_policy/{}'.format(expt_setting))
            probe_input_save_path = os.path.join(temp_name, 'probe_input')
            adv_generate_probe_input_save_path = os.path.join(temp_name, 'probe_input')
            expt_result_save_path  = os.path.join(expt_result_save_path, f'{expt_dataset_name}')
            
            teacher_buffer_save_path = teacher_save_path.replace('teacher_policy', 'teacher_buffer')
            
        else: 
            teacher_buffer_save_path = args.teacher_buffer_save_path
            expt_name = datasets_and_models_set

            project_data_root_path = re.search(f'.+(?=/{expt_name})', teacher_buffer_save_path)[0]
            
            result_root_path = os.path.join(project_data_root_path, expt_name)
            
            expt_result_save_path = os.path.join(project_data_root_path, 'result_save', expt_name)
            temp_name = os.path.join(result_root_path, expt_dataset_name)
            
            if ".zip" in args.teacher_save_path:
                teacher_save_path = args.teacher_save_path
            else:
                teacher_save_path = os.path.join(temp_name, 'teacher_policy/{}'.format(expt_setting))
            

            student_save_path = os.path.join(temp_name, 'student_policy/{}'.format(expt_setting))
            shadow_policy_save_path = os.path.join(temp_name, 'teacher_shadow_policy/{}'.format(expt_setting))
            probe_input_save_path = os.path.join(temp_name, 'probe_input')
            adv_generate_probe_input_save_path = os.path.join(temp_name, 'probe_input')
            expt_result_save_path  = os.path.join(expt_result_save_path, f'{expt_dataset_name}')
            
            # import pdb;pdb.set_trace()



        teacher_train_times = args.teacher_train_times
        teacher_reward_threshold = args.teacher_reward_threshold

        teacher_model_tag = args.teacher_model_tag

        teacher_buffer_length = args.teacher_buffer_length
        teacher_buffer_create_method = args.teacher_buffer_create_method

        num_of_cpu_processor = args.num_of_cpu_processor
        num_of_gpu = args.num_of_gpu ## NOT enable

        mixed_buffer_divided_number = args.mixed_buffer_divided_number
        
        student_agent_type = args.student_agent_type
        teacher_buffer_mix_flag = args.teacher_buffer_mix_flag

        # fingerprint_from_random_sample
        fingerprint_generate_method = args.fingerprint_generate_method
        length_of_single_fingerprint = teacher_buffer_length
        total_number_of_fingerprint = args.total_number_of_fingerprint

        shadow_policy_model_tag = args.shadow_policy_model_tag

        number_episode_optimization = args.number_episode_optimization
        distance_k = args.distance_k
        debug = args.debug
        
        trajectory_size = args.trajectory_size
        trajectory_splitting_num = args.trajectory_splitting_num
        
        significance_level = args.significance_level
        random_seed = args.random_seed

        teacher_buffer_is_mixed_buffer = args.teacher_buffer_is_mixed_buffer
        student_model_tag = args.student_model_tag

        teacher_agent_type = args.teacher_agent_type
        multi_experiment_settings = args.multi_experiment_settings
        save_both_states_and_actions = args.save_both_states_and_actions

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
        cuda_id = 0

        teacher_agent_from = args.teacher_agent_from
        
        num_of_audited_episode = args.num_of_audited_episode
        num_shadow_student = args.num_shadow_student
        critic_model_tag = args.critic_model_tag
        probe_student_output_name = args.probe_student_output_name
        
        
    else:
    ## for debug
        env_name = 'Ant-v2'
        expt_dataset_name = 'sac_ant-v2'
        
        result_root_path = os.getcwd() + '-data/fingerprint_sample_generate_double-check'

        
        temp_name = os.path.join(result_root_path, expt_dataset_name)

        teacher_save_path = os.path.join(temp_name, 'teacher_policy/baseline')
        teacher_buffer_save_path = os.path.join(temp_name, 'teacher_buffer/baseline')
        student_save_path = os.path.join(temp_name, 'student_policy/baseline')
        shadow_policy_save_path = os.path.join(temp_name, 'teacher_shadow_policy/baseline')
        probe_input_save_path = os.path.join(temp_name, 'probe_input')
        adv_generate_probe_input_save_path = os.path.join(temp_name, 'probe_input')

        expt_result_save_path = os.getcwd() + '-data/result_save/fingerprint_sample_generate_double-check'
        expt_result_save_path  = os.path.join(expt_result_save_path, f'{expt_dataset_name}')



        teacher_train_times = 20
        teacher_reward_threshold = 200  # unable

        teacher_model_tag = 'model_500000.pt'

        teacher_buffer_length = 2000000
        teacher_buffer_create_method = 'trained'

        num_of_cpu_processor = 12
        num_of_gpu = 4

        mixed_buffer_divided_number = 2
        
        
        # student_agent_type = 'BC'
        # student_agent_type = 'DiscreteBC'
        # student_agent_type = 'CQL'
        # student_agent_type = 'DiscreteCQL'
        student_agent_type = 'BCQ'
        # student_agent_type = 'DiscreteBCQ'

        teacher_buffer_mix_flag = False

        # fingerprint_from_random_sample
        fingerprint_generate_method = 'fingerprint_from_all_buffer'
        length_of_single_fingerprint = teacher_buffer_length
        total_number_of_fingerprint = 1

        shadow_policy_model_tag = 'model_200000.pt'

        number_episode_optimization = 1000
        distance_k = 0
        debug = True
        
        random_seed = 62

        teacher_buffer_is_mixed_buffer = False
        student_model_tag = 'model_50000.pt'

        teacher_agent_type = 'SAC'
        multi_experiment_settings = True
        save_both_states_and_actions = True

        cuda_id = 1

        teacher_agent_from = 'stable-baselines3'
        num_of_audited_episode = 100
        critic_model_tag = "ckpt_100.pt"
        probe_student_output_name = 'model_SAC_20221017142406518'

    main_exp = MainExperiment(which_experiment,
                              env_name,
                              result_root_path,
                              teacher_save_path,
                              teacher_train_times,
                              teacher_reward_threshold,
                              teacher_model_tag,
                              teacher_buffer_save_path,
                              teacher_buffer_length,
                              teacher_buffer_create_method,
                              num_of_cpu_processor,
                              num_of_gpu,
  
                              student_agent_type,
                              student_save_path,
                              teacher_buffer_mix_flag,
  
                              fingerprint_generate_method,
                              length_of_single_fingerprint,
                              total_number_of_fingerprint,
  
                              shadow_policy_save_path,
                              shadow_policy_model_tag,
                              probe_input_save_path,
                              adv_generate_probe_input_save_path,
                              number_episode_optimization,
                              distance_k,
  
                              mixed_buffer_divided_number,
                              teacher_buffer_is_mixed_buffer,
                              student_model_tag,
                              teacher_agent_type,
                              multi_experiment_settings,
                              save_both_states_and_actions,
                              expt_dataset_name,
                              cuda_id,
                              expt_result_save_path,
                              teacher_agent_from,
                              num_of_audited_episode,
                              num_shadow_student, 
                              critic_model_tag,
  
                              probe_student_output_name,
                              
                              trajectory_size, 
                              trajectory_splitting_num,
                              
                              significance_level,

                              random_seed,
                              debug=debug
                              )
