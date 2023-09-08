# ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning

## Introduction

This is a guideline for reproducing the experiment of NDSS paper named **"ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning"** (https://dx.doi.org/10.14722/ndss.2024.23184). 
This project is licensed under the MIT License. 

For ease of understanding, we divide the workflow into two phases, i.e., preparation and execution. In the preparation, we build the offline datasets and train the offline RL models. Then, in the execution, we utilize the proposed method (ORL-Auditor) to audit the suspect models.

**1. In code and file naming, the `teacher` model is synonymous with the `online` model, and the `student` model and the `offline` model have the same meaning.**

**2. Change the random seed if you need to run one script many times; otherwise, you will get the same model or results.**

**3. The `main.py` is the entrance of all experiments.**

## Run
### Phase 1: Preparation

Firstly, we need to configure the dependencies with the `Dockerfile` before starting the experiments.

```/bin/bash
cd $PROJECT_SAVE_PATH
docker build -t orl-auditor:latest .  ## build the docker image
docker run -it --gpus all  -v $PROJECT_SAVE_PATH:/workspace/off-rl  -d  orl-auditor:latest   /bin/bash   ## start a container
source activate   ## activate the virtualenv (venv)
```

Then, we can advance to the following steps. Table II of the paper illustrates the detailed process and the random seed settings.

<!-- **NOTE:** `phase1_preparation.sh` allows for the automated execution of the original steps in Phase 1. This enhancement enables a straightforward execution of all necessary model preparations with the simple command `bash phase1_preparation.sh`. Additionally, specific experimental settings employed can be reviewed within `phase1_preparation.sh`, spanning from Line 53 to Line 58. -->

#### Step 1: Train online RL models

<!-- ```python
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment train_teacher_model  --teacher_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline  --teacher_train_times 1000000 --random_seed 0  --cuda 0
``` -->
```bash
bash phase1_step1.sh
```

`phase1_step1.sh` will create a new directory `./datasets_and_models_set1` once it is finished.

```python
./datasets_and_models_set1
├── logs
│   └── sac_lunarlander
│       └── train_teacher_model
│           ├── 0.txt
│           └── 1.txt
└── sac_lunarlander
    └── teacher_policy
        └── baseline
            ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip
            └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip

6 directories, 4 files
```

<!-- #### Step 2: Evaluate the online RL models

Here, we evaluate the performance of the online model named `sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000.zip`. 

```python
python main.py  --env_name LunarLanderContinuous-v2 --which_experiment eval_teacher_model  --teacher_save_path ./datasets_and_models_set1/sac_lunarlander/teacher_policy/baseline/sb3_sac_LunarLanderContinuous-v2_20221017163911171_1000000.zip  --cuda 0
``` -->

#### Step 2: Create the offline datasets

```bash
bash phase1_step2.sh
```

`phase1_step2.sh` will save the offline datasets in `./datasets_and_models_set1/sac_lunarlander/teacher_buffer/` once it is finished.

```python
./datasets_and_models_set1
├── logs
│   └── sac_lunarlander
│       ├── teacher_buffer
│       │   ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip.txt
│       │   └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip.txt
│       └── train_teacher_model
│           ├── 0.txt
│           └── 1.txt
└── sac_lunarlander
    ├── teacher_buffer
    │   └── baseline
    │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5
    │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5
    └── teacher_policy
        └── baseline
            ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip
            └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip

9 directories, 8 files
```

#### Step 3: Train offline RL models

```bash
bash phase1_step3.sh
```

`phase1_step3.sh` will save the trained offline RL models in `./datasets_and_models_set1/sac_lunarlander/student_policy/` once it is finished. 

```python
./datasets_and_models_set1
├── logs
│   └── sac_lunarlander
│       ├── teacher_buffer
│       │   ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip.txt
│       │   └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip.txt
│       ├── train_student_model
│       │   └── BC
│       │       ├── 0.txt
│       │       ├── 1.txt
│       │       ├── 2.txt
│       │       ├── 3.txt
│       │       └── 4.txt
│       └── train_teacher_model
│           ├── 0.txt
│           └── 1.txt
└── sac_lunarlander
    ├── student_policy
    │   └── baseline
    │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5
    │       │   ├── BC_20230908122143
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122145
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122146
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122148
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122149
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   └── runs
    │       │       ├── BC_20230908122143
    │       │       │   ├── BC_20230908122143
    │       │       │   │   ├── events.out.tfevents.1694161342.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161379.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161417.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161453.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161489.608dbb339466
    │       │       │   └── events.out.tfevents.1694161303.608dbb339466
    │       │       ├── BC_20230908122145
    │       │       │   ├── BC_20230908122145
    │       │       │   │   ├── events.out.tfevents.1694161343.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161381.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161419.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161455.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161492.608dbb339466
    │       │       │   └── events.out.tfevents.1694161305.608dbb339466
    │       │       ├── BC_20230908122146
    │       │       │   ├── BC_20230908122146
    │       │       │   │   ├── events.out.tfevents.1694161341.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161374.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161406.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161439.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161472.608dbb339466
    │       │       │   └── events.out.tfevents.1694161306.608dbb339466
    │       │       ├── BC_20230908122148
    │       │       │   ├── BC_20230908122148
    │       │       │   │   ├── events.out.tfevents.1694161351.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161393.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161435.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161476.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161516.608dbb339466
    │       │       │   └── events.out.tfevents.1694161308.608dbb339466
    │       │       └── BC_20230908122149
    │       │           ├── BC_20230908122149
    │       │           │   ├── events.out.tfevents.1694161353.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161395.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161439.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161480.608dbb339466
    │       │           │   └── events.out.tfevents.1694161518.608dbb339466
    │       │           └── events.out.tfevents.1694161309.608dbb339466
    │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5
    │           ├── BC_20230908122150
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122152
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122154
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122155
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122156
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           └── runs
    │               ├── BC_20230908122150
    │               │   ├── BC_20230908122150
    │               │   │   ├── events.out.tfevents.1694161355.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161398.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161441.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161483.608dbb339466
    │               │   │   └── events.out.tfevents.1694161521.608dbb339466
    │               │   └── events.out.tfevents.1694161310.608dbb339466
    │               ├── BC_20230908122152
    │               │   ├── BC_20230908122152
    │               │   │   ├── events.out.tfevents.1694161351.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161388.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161424.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161461.608dbb339466
    │               │   │   └── events.out.tfevents.1694161496.608dbb339466
    │               │   └── events.out.tfevents.1694161312.608dbb339466
    │               ├── BC_20230908122154
    │               │   ├── BC_20230908122154
    │               │   │   ├── events.out.tfevents.1694161349.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161382.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161414.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161447.608dbb339466
    │               │   │   └── events.out.tfevents.1694161480.608dbb339466
    │               │   └── events.out.tfevents.1694161314.608dbb339466
    │               ├── BC_20230908122155
    │               │   ├── BC_20230908122155
    │               │   │   ├── events.out.tfevents.1694161359.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161402.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161444.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161485.608dbb339466
    │               │   │   └── events.out.tfevents.1694161523.608dbb339466
    │               │   └── events.out.tfevents.1694161315.608dbb339466
    │               └── BC_20230908122156
    │                   ├── BC_20230908122156
    │                   │   ├── events.out.tfevents.1694161360.608dbb339466
    │                   │   ├── events.out.tfevents.1694161403.608dbb339466
    │                   │   ├── events.out.tfevents.1694161445.608dbb339466
    │                   │   ├── events.out.tfevents.1694161487.608dbb339466
    │                   │   └── events.out.tfevents.1694161525.608dbb339466
    │                   └── events.out.tfevents.1694161316.608dbb339466
    ├── teacher_buffer
    │   └── baseline
    │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5
    │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5
    └── teacher_policy
        └── baseline
            ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip
            └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip

47 directories, 173 files
```

#### Step 4: Train critic model


```bash
bash phase1_step4.sh
```

`phase1_step4.sh` will save the trained critic models in `./datasets_and_models_set1/sac_lunarlander/auditor/` once it is finished.

```python
./datasets_and_models_set1
├── logs
│   └── sac_lunarlander
│       ├── auditor
│       │   ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5.txt
│       │   └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5.txt
│       ├── teacher_buffer
│       │   ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip.txt
│       │   └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip.txt
│       ├── train_student_model
│       │   └── BC
│       │       ├── 0.txt
│       │       ├── 1.txt
│       │       ├── 2.txt
│       │       ├── 3.txt
│       │       └── 4.txt
│       └── train_teacher_model
│           ├── 0.txt
│           └── 1.txt
└── sac_lunarlander
    ├── auditor
    │   ├── teacher_buffer_in_transition_form
    │   │   └── baseline
    │   │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5.npy
    │   │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5.npy
    │   └── trained_critic_model
    │       └── baseline
    │           ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000
    │           │   ├── ckpt_100.pt
    │           │   ├── ckpt_10.pt
    │           │   ├── ckpt_110.pt
    │           │   ├── ckpt_120.pt
    │           │   ├── ckpt_130.pt
    │           │   ├── ckpt_140.pt
    │           │   ├── ckpt_150.pt
    │           │   ├── ckpt_160.pt
    │           │   ├── ckpt_170.pt
    │           │   ├── ckpt_180.pt
    │           │   ├── ckpt_190.pt
    │           │   ├── ckpt_200.pt
    │           │   ├── ckpt_20.pt
    │           │   ├── ckpt_30.pt
    │           │   ├── ckpt_40.pt
    │           │   ├── ckpt_50.pt
    │           │   ├── ckpt_60.pt
    │           │   ├── ckpt_70.pt
    │           │   ├── ckpt_80.pt
    │           │   └── ckpt_90.pt
    │           └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000
    │               ├── ckpt_100.pt
    │               ├── ckpt_10.pt
    │               ├── ckpt_110.pt
    │               ├── ckpt_120.pt
    │               ├── ckpt_130.pt
    │               ├── ckpt_140.pt
    │               ├── ckpt_150.pt
    │               ├── ckpt_160.pt
    │               ├── ckpt_170.pt
    │               ├── ckpt_180.pt
    │               ├── ckpt_190.pt
    │               ├── ckpt_200.pt
    │               ├── ckpt_20.pt
    │               ├── ckpt_30.pt
    │               ├── ckpt_40.pt
    │               ├── ckpt_50.pt
    │               ├── ckpt_60.pt
    │               ├── ckpt_70.pt
    │               ├── ckpt_80.pt
    │               └── ckpt_90.pt
    ├── student_policy
    │   └── baseline
    │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5
    │       │   ├── BC_20230908122143
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122145
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122146
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122148
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   ├── BC_20230908122149
    │       │   │   ├── loss.csv
    │       │   │   ├── model_10000.pt
    │       │   │   ├── model_20000.pt
    │       │   │   ├── model_30000.pt
    │       │   │   ├── model_40000.pt
    │       │   │   ├── model_50000.pt
    │       │   │   ├── params.json
    │       │   │   ├── time_algorithm_update.csv
    │       │   │   ├── time_sample_batch.csv
    │       │   │   └── time_step.csv
    │       │   └── runs
    │       │       ├── BC_20230908122143
    │       │       │   ├── BC_20230908122143
    │       │       │   │   ├── events.out.tfevents.1694161342.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161379.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161417.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161453.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161489.608dbb339466
    │       │       │   └── events.out.tfevents.1694161303.608dbb339466
    │       │       ├── BC_20230908122145
    │       │       │   ├── BC_20230908122145
    │       │       │   │   ├── events.out.tfevents.1694161343.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161381.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161419.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161455.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161492.608dbb339466
    │       │       │   └── events.out.tfevents.1694161305.608dbb339466
    │       │       ├── BC_20230908122146
    │       │       │   ├── BC_20230908122146
    │       │       │   │   ├── events.out.tfevents.1694161341.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161374.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161406.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161439.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161472.608dbb339466
    │       │       │   └── events.out.tfevents.1694161306.608dbb339466
    │       │       ├── BC_20230908122148
    │       │       │   ├── BC_20230908122148
    │       │       │   │   ├── events.out.tfevents.1694161351.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161393.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161435.608dbb339466
    │       │       │   │   ├── events.out.tfevents.1694161476.608dbb339466
    │       │       │   │   └── events.out.tfevents.1694161516.608dbb339466
    │       │       │   └── events.out.tfevents.1694161308.608dbb339466
    │       │       └── BC_20230908122149
    │       │           ├── BC_20230908122149
    │       │           │   ├── events.out.tfevents.1694161353.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161395.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161439.608dbb339466
    │       │           │   ├── events.out.tfevents.1694161480.608dbb339466
    │       │           │   └── events.out.tfevents.1694161518.608dbb339466
    │       │           └── events.out.tfevents.1694161309.608dbb339466
    │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5
    │           ├── BC_20230908122150
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122152
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122154
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122155
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           ├── BC_20230908122156
    │           │   ├── loss.csv
    │           │   ├── model_10000.pt
    │           │   ├── model_20000.pt
    │           │   ├── model_30000.pt
    │           │   ├── model_40000.pt
    │           │   ├── model_50000.pt
    │           │   ├── params.json
    │           │   ├── time_algorithm_update.csv
    │           │   ├── time_sample_batch.csv
    │           │   └── time_step.csv
    │           └── runs
    │               ├── BC_20230908122150
    │               │   ├── BC_20230908122150
    │               │   │   ├── events.out.tfevents.1694161355.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161398.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161441.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161483.608dbb339466
    │               │   │   └── events.out.tfevents.1694161521.608dbb339466
    │               │   └── events.out.tfevents.1694161310.608dbb339466
    │               ├── BC_20230908122152
    │               │   ├── BC_20230908122152
    │               │   │   ├── events.out.tfevents.1694161351.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161388.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161424.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161461.608dbb339466
    │               │   │   └── events.out.tfevents.1694161496.608dbb339466
    │               │   └── events.out.tfevents.1694161312.608dbb339466
    │               ├── BC_20230908122154
    │               │   ├── BC_20230908122154
    │               │   │   ├── events.out.tfevents.1694161349.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161382.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161414.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161447.608dbb339466
    │               │   │   └── events.out.tfevents.1694161480.608dbb339466
    │               │   └── events.out.tfevents.1694161314.608dbb339466
    │               ├── BC_20230908122155
    │               │   ├── BC_20230908122155
    │               │   │   ├── events.out.tfevents.1694161359.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161402.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161444.608dbb339466
    │               │   │   ├── events.out.tfevents.1694161485.608dbb339466
    │               │   │   └── events.out.tfevents.1694161523.608dbb339466
    │               │   └── events.out.tfevents.1694161315.608dbb339466
    │               └── BC_20230908122156
    │                   ├── BC_20230908122156
    │                   │   ├── events.out.tfevents.1694161360.608dbb339466
    │                   │   ├── events.out.tfevents.1694161403.608dbb339466
    │                   │   ├── events.out.tfevents.1694161445.608dbb339466
    │                   │   ├── events.out.tfevents.1694161487.608dbb339466
    │                   │   └── events.out.tfevents.1694161525.608dbb339466
    │                   └── events.out.tfevents.1694161316.608dbb339466
    ├── teacher_buffer
    │   └── baseline
    │       ├── sb3_sac_LunarLanderContinuous-v2_0_10000-50000.h5
    │       └── sb3_sac_LunarLanderContinuous-v2_1_10000-50000.h5
    └── teacher_policy
        └── baseline
            ├── sb3_sac_LunarLanderContinuous-v2_0_10000.zip
            └── sb3_sac_LunarLanderContinuous-v2_1_10000.zip

55 directories, 217 files
```

### Phase 2: Execution and draw

**NOTE:** `phase2_execution.sh` serves to audit the datasets and calculates the TPR/TNR values. These values, as demonstrated in TABLE IV, Fig. 6, Fig. 7, and Fig. 8, effectively showcase the effectiveness of ORL-Auditor.
`phase2_draw.sh` can convert the raw results into a human-readable table.
Modifications to hyperparameter settings can also be easily implemented within `experimental_settings.yml`.

```bash
bash phase2_execution.sh
```

`phase2_execution.sh` will save the TPR/TNR values in `./result_save/datasets_and_models_set1/sac_lunarlander/sac_lunarlander` once it is finished.

```python
./result_save
└── datasets_and_models_set1
    └── sac_lunarlander
        ├── audit_result-numepi_5-envname_LunarLanderContinuous-v2-critag_ckpt_200-sustype_BC-sustag_model_50000.pt-numstu_3-trajsize_1.0-signlevel_0.01-20230908124624549.json
        └── audit_result-numepi_5-envname_LunarLanderContinuous-v2-critag_ckpt_200-sustype_BC-sustag_model_50000.pt-numstu_3-trajsize_1.0-signlevel_0.01-20230908124624549.xlsx

2 directories, 2 files
```

```bash
bash phase2_draw.sh
```

The human-readable table is saved as `./result_save/datasets_and_models_set1/audit_results-numstu_3-trajsize_1.0-signlevel_0.01.xlsx`. The file's name contain the exact hyperparameter settings, e.g., num_shadow_student, trajectory_size, and significance_level. 

The main results in the paper and the corresponding hyperparameter settings are shown below.

#### Results in Table IV

```ymal
num_shadow_student: 15
significance_level: 0.01
trajectory_size: 1.0
```

#### Results in Fig. 6

9 shadow models

```ymal
num_shadow_student: 9
significance_level: 0.01
trajectory_size: 1.0
```

21 shadow models

```ymal
num_shadow_student: 21
significance_level: 0.01
trajectory_size: 1.0
```

#### Results in Fig. 7

Significance level = 0.001

```ymal
num_shadow_student: 15
significance_level: 0.001
trajectory_size: 1.0
```

Significance level = 0.0001

```ymal
num_shadow_student: 15
significance_level: 0.0001
trajectory_size: 1.0
```

#### Results in Fig. 8

25% of full trajectory

```bash
num_shadow_student: 15
significance_level: 0.01
trajectory_size: 0.25
```

50% of full trajectory

```bash
num_shadow_student: 15
significance_level: 0.01
trajectory_size: 0.5
```

## Citation
```
@inproceedings{DCSJCCZ24,
  author = {Linkang Du and Min Chen and Mingyang Sun and Shouling Ji and Peng Cheng and Jiming Chen and Zhikun Zhang},
  booktitle = {{Network and Distributed System Security Symposium (NDSS)}},
  publisher = {Internet Society},
  title = {{ORL-Auditor: Dataset Auditing in Offline Deep Reinforcement Learning}},
  year = {2024},
}
```
