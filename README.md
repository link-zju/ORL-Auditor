# ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning

## Introduction

This is a guideline for reproducing the experiment of NDSS paper named **"ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning"** (https://dx.doi.org/10.14722/ndss.2024.23184). 
This project is licensed under the MIT License. 

For ease of understanding, we divide the workflow into two phases, i.e., preparation and execution. In the preparation, we build the offline datasets and train the offline RL models. Then, in the execution, we utilize the proposed method (ORL-Auditor) to audit the suspect models.

**1. In code and file naming, the `teacher` model is synonymous with the `online` model, and the `student` model and the `offline` model have the same meaning.**

**2. Change the random seed if you need to run one script many times; otherwise, you will get the same model or results.**

### Create the file directory

We should first create a directory for saving the models and datasets. In the following, we use the `Lunar Lander` task as an example to show the directory.

The `datasets_and_models` directory saves the offline datasets, RL models, critic models, and other intermediate files.
The `result_save` directory saves the experimental results mainly in `.xlsx` format, distinguished by the files' name.
The `lib_util` directory saves the essential functions.

```python
ORL_Auditor_proj
├── datasets_and_models_set1
│   └── sac_lunarlander
│       ├── auditor
│       │   ├── teacher_buffer_in_transition_form
│       │   └── trained_critic_model
│       ├── student_policy
│       │   └── baseline
│       ├── teacher_buffer
│       │   └── baseline
│       └── teacher_policy
│           └── baseline
├── lib_util
└── result_save
    └── datasets_and_models_set1
        └── sac_lunarlander
```

**The `main.py` is the entrance of all experiments.**

### Phase 1: Preparation

Firstly, we need to configure the dependencies with the `Dockerfile` before starting the experiments.

```/bin/bash
cd $PROJECT_SAVE_PATH
docker build -t orl-auditor:latest .  ## build the docker image
docker run -it --gpus all  -v $PROJECT_SAVE_PATH:/workspace/off-rl  -d  orl-auditor:latest   /bin/bash   ## start a container
source activate   ## activate the virtualenv (venv)
```

Then, we can advance to the following steps. Table II of the paper illustrates the detailed process and the random seed settings.

**NOTE:** `phase1_preparation.sh` allows for the automated execution of the original steps in Phase 1. This enhancement enables a straightforward execution of all necessary model preparations with the simple command `bash phase1_preparation.sh`. Additionally, specific experimental settings employed can be reviewed within `phase1_preparation.sh`, spanning from Line 53 to Line 58.

#### Step 1: Train online RL models

```python
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_teacher_model  --teacher_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline  --teacher_train_times 1000000 --random_seed 0  --cuda 0
# Output Directory: ORL_Auditor_proj/datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline
```

#### Step 2: Evaluate the online RL models

Here, we evaluate the performance of the online model named `sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000.zip`. 

```python
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment eval_teacher_model  --teacher_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000.zip  --cuda 0
```

#### Step 3: Create the offline datasets

```python
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment teacher_buffer_create --teacher_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000.zip --teacher_buffer_length 50000 --random_seed 0 --cuda 0
# Output Directory: ORL_Auditor_proj/datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline
```

#### Step 4: Train offline RL models

```python
# Train BC model
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_student_model --student_agent_type  BC  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5 --random_seed 0 --cuda 0

# Train BCQ model
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_student_model --student_agent_type  BCQ  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5 --random_seed 0 --cuda 0

# Train IQL model
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_student_model --student_agent_type  IQL  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5 --random_seed 0 --cuda 0

# Train TD3PlusBC model
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_student_model --student_agent_type  TD3PlusBC  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5 --random_seed 0 --cuda 0

# Output Directory: ORL_Auditor_proj/datasets_and_models_set1/sac_lunarlander/student_policy/baseline
```

The above script trains one offline RL model in every execution. 

#### Step 5: Evaluate offline RL models

```python
# Evalute TD3PlusBC model
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment eval_student_model --student_agent_type  TD3PlusBC  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5 --student_model_tag model_50000.pt --random_seed 0 --cuda 0
# Output Directory: ORL_Auditor_proj/result_save/datasets_and_models_set1/sac_lunarlander
```

#### Step 6: Train critic model

```python
python main.py --env_name LunarLanderContinuous-v2 --which_experiment auditor_train_critic_model  --teacher_buffer_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_buffer/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000-50000.h5  --random_seed 0 --cuda 0
# Output Directory: ORL_Auditor_proj/datasets_and_models_set1/sac_lunarlander/auditor/trained_critic_model/baseline
```

### Phase 2: Execution

**NOTE:** `phase2_execution.sh` serves to audit the datasets and calculates the TPR/TNR values. These values, as demonstrated in TABLE IV, Fig. 6, Fig. 7, and Fig. 8, effectively showcase the effectiveness of ORL-Auditor. Modifications to hyperparameter settings can also be easily implemented within `phase2_execution.sh`, found between Line 43 and Line 46.

#### Results in Table IV

```bash
num_shadow_student=15
significance_level=0.01
trajectory_size=1.0
```

#### Results in Fig. 6

9 shadow models

```bash
num_shadow_student=9
significance_level=0.01
trajectory_size=1.0
```

21 shadow models

```bash
num_shadow_student=21
significance_level=0.01
trajectory_size=1.0
```

#### Results in Fig. 7

significance level = 0.001

```bash
num_shadow_student=15
significance_level=0.001
trajectory_size=1.0
```

significance level = 0.0001

```bash
num_shadow_student=15
significance_level=0.0001
trajectory_size=1.0
```

#### Results in Fig. 8

25% of full trajectory

```bash
num_shadow_student=15
significance_level=0.01
trajectory_size=0.25
```

50% of full trajectory

```bash
num_shadow_student=15
significance_level=0.01
trajectory_size=0.5
```
