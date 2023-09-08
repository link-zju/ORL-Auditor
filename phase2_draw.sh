#!/bin/bash
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

echo $env_name
echo $all_student_model_type
# put the results into a table
python ./lib_util/draw_table.py --env_name_list "$env_name" --suspect_model_type_list "$all_student_model_type" --json_data_dir result_save/$datasets_and_models_dir --num_shadow_student $num_shadow_student --significance_level $significance_level --trajectory_size $trajectory_size

