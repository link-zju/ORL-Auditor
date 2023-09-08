import os 
import gym
import stable_baselines3 as sb3
from lib_util.file_tag import get_time_stamp

class TeacherModelTrain():
    def __init__(self,                 
                train_env, 
                teacher_save_path, 
                teacher_train_times, 
                random_seed,
                cuda_id) -> None:

        self.train_env = train_env
        self.teacher_save_path = teacher_save_path
        self.teacher_train_times = teacher_train_times
        self.random_seed = random_seed
        self.cuda_id = cuda_id
        
        if "LunarLanderContinuous-v2" == self.train_env:
            self.train_sac()
            
        elif "BipedalWalker-v3" == self.train_env:
            self.train_ppo()
            
        elif "Ant-v2" == self.train_env:
            self.train_sac()
             
        else: print("Error: Env setting in teacher_train.py")


    def train_sac(self):
        train_env = gym.make(self.train_env)
        train_env.seed(self.random_seed)
        
        sb3sac = sb3.SAC("MlpPolicy", train_env, verbose=1)
        sb3sac.learn(self.teacher_train_times)
        model_name = f"sb3_sac_{self.train_env}_{self.random_seed}_{self.teacher_train_times}.zip"
        
        if not os.path.exists(self.teacher_save_path):
            print(self.teacher_save_path)
            os.makedirs(self.teacher_save_path)
        model_save_path = os.path.join(self.teacher_save_path, model_name)
        sb3sac.save(model_save_path)
        
        
    def train_ppo(self):
        train_env = gym.make(self.train_env)
        train_env.seed(self.random_seed)
        
        sb3ppo = sb3.PPO("MlpPolicy", train_env, verbose=1)
        sb3ppo.learn(self.teacher_train_times)
        model_name = f"sb3_ppo_{self.train_env}_{self.random_seed}_{self.teacher_train_times}.zip"
        
        if not os.path.exists(self.teacher_save_path):
            os.makedirs(self.teacher_save_path)
            print(self.teacher_save_path)
        model_save_path = os.path.join(self.teacher_save_path, model_name)
        sb3ppo.save(model_save_path)
