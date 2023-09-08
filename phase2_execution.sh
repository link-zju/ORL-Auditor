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
            echo "Waiting 5s for the CPU or GPU resources"
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

datasets_and_models_dir=$(yq -r '.datasets_and_models_dir' "$yaml_file")
number_teacher_model=$(yq -r '.number_teacher_model' "$yaml_file")
teacher_train_times=$(yq -r '.teacher_train_times' "$yaml_file")
number_student_model=$(yq -r '.number_student_model' "$yaml_file")
env_name=$(yq -r '.env_name[]' "$yaml_file")
all_student_model_type=$(yq -r '.all_student_model_type[]' "$yaml_file")

num_shadow_student=$(yq -r '.num_shadow_student' "$yaml_file")
significance_level=$(yq -r '.significance_level' "$yaml_file")
trajectory_size=$(yq -r '.trajectory_size' "$yaml_file")
random_seed=$(yq -r '.random_seed' "$yaml_file")
critic_model_tag=$(yq -r '.critic_model_tag' "$yaml_file")
student_model_tag=$(yq -r '.student_model_tag' "$yaml_file")
num_of_audited_episode=$(yq -r '.num_of_audited_episode' "$yaml_file")


if [ ! -d "result_save" ]; then
  mkdir "result_save"
fi

for task_name in ${env_name[@]};
do
    for student_model_type in ${all_student_model_type[@]};
    do
    select_GPU
    echo $task_name $student_model_type
    mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type"
    touch ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type/$random_seed.txt
    python main.py --datasets_and_models_dir $datasets_and_models_dir --env_name $task_name --which_experiment audit_dataset  --teacher_buffer_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline --critic_model_tag $critic_model_tag --student_model_tag $student_model_tag --student_agent_type  $student_model_type  --num_of_audited_episode $num_of_audited_episode --num_shadow_student $num_shadow_student --significance_level $significance_level  --trajectory_size $trajectory_size --random_seed $random_seed --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type/$random_seed.txt &
    done
done
