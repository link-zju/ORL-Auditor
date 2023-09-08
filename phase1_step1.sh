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
datasets_and_models_dir=$(awk '/datasets_and_models_dir:/ {print $2}' $yaml_file)
echo "datasets_and_models_dir: $datasets_and_models_dir"
number_teacher_model=$(awk '/number_teacher_model:/ {print $2}' $yaml_file)
echo "number_teacher_model: $number_teacher_model"
teacher_train_times=$(awk '/teacher_train_times:/ {print $2}' $yaml_file)
echo "teacher_train_times: $teacher_train_times"
number_student_model=$(awk '/number_student_model:/ {print $2}' $yaml_file)
echo "number_student_model: $number_student_model"

env_name=$(yq -r '.env_name[]' "$yaml_file")
echo "env_name: $env_name"
all_student_model_type=$(yq -r '.all_student_model_type[]' "$yaml_file")
echo "all_student_model_type: $all_student_model_type"


if [ ! -d "$datasets_and_models_dir" ]; then
  mkdir "$datasets_and_models_dir"
fi

for task_name in ${env_name[@]};
do
    echo $task_name
    for i in $(seq 0 $[$number_teacher_model-1]);
    do
        select_GPU
        mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model"
        touch ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model/$i.txt
        python main.py  --datasets_and_models_dir $datasets_and_models_dir --env_name $task_name --which_experiment train_teacher_model  --teacher_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline  --teacher_train_times $teacher_train_times --random_seed $i  --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model/$i.txt &
    done 
done
