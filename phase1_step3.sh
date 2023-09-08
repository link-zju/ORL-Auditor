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
    ## train offline policy
    teacher_buffer_name=$(ls ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline | grep ".h" )
    for buffer_name in $teacher_buffer_name;
    do 
        for student_model_type in ${all_student_model_type[@]};
        do
            for i in $(seq 0 $[$number_student_model-1]);
            do
            select_GPU
            echo $buffer_name $student_model_type $i
            mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_student_model/$student_model_type"
            touch ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_student_model/$student_model_type/$i.txt
            python main.py  --datasets_and_models_dir $datasets_and_models_dir  --env_name $task_name --which_experiment train_student_model --student_agent_type  $student_model_type  --teacher_buffer_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline/$buffer_name --random_seed $i  --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_student_model/$student_model_type/$i.txt &
            done
        done
    done 
done
