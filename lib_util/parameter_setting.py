import argparse
def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def offlineRL_main_exp():  # intialize paprameters
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--env_name", type=str, default='LunarLanderContinuous-v2')
    parser.add_argument("--which_experiment", type=str, default='')
    parser.add_argument("--teacher_buffer_save_path", type=str, default='')
    

    parser.add_argument("--teacher_train_times", type=int, default=10)
    parser.add_argument("--teacher_reward_threshold", type=int, default=500)

    
    parser.add_argument("--teacher_model_tag", type=str, default='model_500000.pt')

    parser.add_argument("--teacher_buffer_length", type=int, default=500000)
    parser.add_argument("--teacher_buffer_create_method", type=str, default='trained')

    parser.add_argument("--num_of_cpu_processor", type=int, default=12)
    parser.add_argument("--num_of_gpu", type=int, default=8)
    parser.add_argument("--mixed_buffer_divided_number", type=int, default=2)
    
    
    parser.add_argument("--student_agent_type", type=str, default='BC')
    parser.add_argument("--teacher_buffer_mix_flag", type=str2bool, default=False)
    
    parser.add_argument("--fingerprint_generate_method", type=str, default='fingerprint_from_all_buffer')
    parser.add_argument("--total_number_of_fingerprint", type=int, default=1)

    parser.add_argument("--shadow_policy_model_tag", type=str, default='model_200000.pt')
    
    
    parser.add_argument("--number_episode_optimization", type=int, default=1000)
    parser.add_argument("--distance_k", type=float, default=0)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=0)
    
    parser.add_argument("--trajectory_size", type=float, default=1.0)
    parser.add_argument("--trajectory_splitting_num", type=int, default=1)
    
    
    
    parser.add_argument("--teacher_buffer_is_mixed_buffer", type=str2bool, default=False)
    

    parser.add_argument("--student_model_tag", type=str, default='model_250000.pt')
    parser.add_argument("--teacher_agent_type", type=str, default='SAC')

    parser.add_argument("--multi_experiment_settings", type=str2bool, default=True)
    parser.add_argument("--save_both_states_and_actions", type=str2bool, default=True)

    parser.add_argument("--cuda", type=int, default=0)
    
    parser.add_argument("--teacher_agent_from", type=str, default='stable-baselines3')
    parser.add_argument("--teacher_save_path", type=str, default='')
    parser.add_argument("--num_of_audited_episode", type=int, default=5)
    parser.add_argument("--num_shadow_student", type=int, default=7)
    
    parser.add_argument("--critic_model_tag", type=str, default='ckpt_100.pt')
    parser.add_argument("--probe_student_output_name", type=str, default='model_SAC_20221017142406518')
    
    
    args = parser.parse_args()
    print(args)

    return args