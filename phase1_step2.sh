#!/bin/bash

## GPU workload balance 
gpu_id=-1
select_GPU()
{
    while :
    do
        gpu_id=`python3 -c 'import lib_util.select_gpu as sg; sg.select_gpu()'`
        if [ "$gpu_id" = "-1" ]; 
        then
            sleep 5
            continue
        else
            break
        fi
    done
}

## name transfer
declare -A env_name_to_expt_dataset_name
env_name_to_expt_dataset_name=([LunarLanderContinuous-v2]=sac_lunarlander \
                               [BipedalWalker-v3]=sac_bipedalwalker \
                               [Ant-v2]=sac_ant)

yaml_file="experimental_settings.yml"
# datasets_and_models_dir=$(awk '/datasets_and_models_dir:/ {print $2}' $yaml_file)
datasets_and_models_dir=$(yq -r '.datasets_and_models_dir' "$yaml_file")
number_teacher_model=$(yq -r '.number_teacher_model' "$yaml_file")
teacher_train_times=$(yq -r '.teacher_train_times' "$yaml_file")
number_student_model=$(yq -r '.number_student_model' "$yaml_file")
env_name=$(yq -r '.env_name[]' "$yaml_file")
all_student_model_type=$(yq -r '.all_student_model_type[]' "$yaml_file")

for task_name in ${env_name[@]};
do
    echo $task_name
    echo $datasets_and_models_dir
    # Collect datasets
    teacher_model_name=$(ls ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline | grep ".zip")
    for model_name in $teacher_model_name;
    do
        select_GPU
        mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer"
        python main.py  --datasets_and_models_dir $datasets_and_models_dir  --env_name $task_name --which_experiment teacher_buffer_create --teacher_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline/$model_name --teacher_buffer_length 50000 --random_seed 0 --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/$model_name.txt &
    done
done
