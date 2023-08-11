import torch
from my_auditor import CriticModelStateActionValueTest1

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')

critic_model_path = '../fingerprint_sample_generate_double-check/dqn_cartpole_copy/auditor/trained_critic_model/test1/model_SAC_20221017142406518-500000/ckpt_50.pt'
new_critic_model_path = '../fingerprint_sample_generate_double-check/dqn_cartpole_copy/auditor/trained_critic_model/test1/model_SAC_20221017142406518-500000/ckpt_50_new.pt'
observation_action_size = 10
critic_model = CriticModelStateActionValueTest1(observation_action_size, n_hidden=128, n_output=1).to(device)
critic_model.load_state_dict(torch.load(critic_model_path)['model_state_dict'])
torch.save({'model_state_dict': critic_model.state_dict()}, new_critic_model_path)