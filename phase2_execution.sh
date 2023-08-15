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

## Check if the previous tasks is finished
forward_task_complete()
{
    ongoing_process=`ps aux|grep processname|grep "main.py"|wc -l`
    while [ $ongoing_process -gt 0 ]
    do 
        sleep 5
        echo "Waiting for the previous task to finish"
        ongoing_process=`ps aux|grep processname|grep "main.py"|wc -l`
    done

}



## name transfer
declare -A env_name_to_expt_dataset_name
env_name_to_expt_dataset_name=([LunarLanderContinuous-v2]=sac_lunarlander \
                               [BipedalWalker-v3]=sac_bipedalwalker \
                               [Ant-v2]=sac_ant)

## default settings in the paper's evaluation
num_shadow_student=15
significance_level=0.01
trajectory_size=1.0
random_seed=0


datasets_and_models_dir=datasets_and_models_set1

env_name=("LunarLanderContinuous-v2" "BipedalWalker-v3" "Ant-v2")
all_student_model_type=("BC" "BCQ" "IQL" "TD3PlusBC")

# for task_name in ${env_name[@]};
# do
#     for student_model_type in ${all_student_model_type[@]};
#     do
#     select_GPU
#     echo $task_name $student_model_type
#     mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type"
#     touch ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type/$random_seed.txt
#     python main.py --datasets_and_models_dir $datasets_and_models_dir --env_name $task_name --which_experiment audit_dataset  --teacher_buffer_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline --critic_model_tag ckpt_200.pt --student_model_tag model_50000.pt --student_agent_type  $student_model_type  --num_of_audited_episode 2 --num_shadow_student $num_shadow_student --significance_level $significance_level  --trajectory_size $trajectory_size --random_seed $random_seed --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/data_audit/$student_model_type/$random_seed.txt &
#     done
# done
forward_task_complete
## put the results into a table
python ./lib_util/draw_table.py --json_data_dir result_save/$datasets_and_models_dir --num_shadow_student $num_shadow_student --significance_level $significance_level --trajectory_size $trajectory_size

