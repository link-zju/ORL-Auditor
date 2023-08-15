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


forward_task_complete
## name transfer
declare -A env_name_to_expt_dataset_name
env_name_to_expt_dataset_name=([LunarLanderContinuous-v2]=sac_lunarlander \
                               [BipedalWalker-v3]=sac_bipedalwalker \
                               [Ant-v2]=sac_ant)



## fast evaluation config
# datasets_and_models_dir=datasets_and_models_set1
# number_teacher_model=2
# env_name=("LunarLanderContinuous-v2")
# teacher_train_times=10000
# number_student_model=2
# all_student_model_type=("BC" "BCQ" "IQL" "TD3PlusBC")


## default settings in the paper's evaluation
datasets_and_models_dir=datasets_and_models_set1
number_teacher_model=5
env_name=("LunarLanderContinuous-v2" "BipedalWalker-v3" "Ant-v2")
teacher_train_times=1000000
number_student_model=30
all_student_model_type=("BC" "BCQ" "IQL" "TD3PlusBC")





for task_name in ${env_name[@]};
do
    # train online policy
    for i in $(seq 0 $[$number_teacher_model-1]);
    do
        select_GPU
        mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model"
        touch ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model/$i.txt
        python main.py  --datasets_and_models_dir $datasets_and_models_dir --env_name $task_name --which_experiment train_teacher_model  --teacher_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline  --teacher_train_times $teacher_train_times --random_seed $i  --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/train_teacher_model/$i.txt &
    done 

    forward_task_complete

    # collect datasets
    teacher_model_name=$(ls ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline | grep ".zip")
    echo $teacher_model_name
    for model_name in $teacher_model_name;
    do
        select_GPU
        mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer"
        python main.py  --datasets_and_models_dir $datasets_and_models_dir  --env_name $task_name --which_experiment teacher_buffer_create --teacher_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_policy/baseline/$model_name --teacher_buffer_length 50000 --random_seed 0 --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/$model_name.txt &
    done

    forward_task_complete

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

    forward_task_complete

    ## Train critic model
    teacher_buffer_name=$(ls ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline | grep ".h")
    echo $teacher_buffer_name
    mkdir -p "./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/auditor/teacher_buffer_in_transition_form/baseline"
    for buffer_name in $teacher_buffer_name;
    do
        select_GPU
        mkdir -p "./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/auditor"
        python main.py  --datasets_and_models_dir $datasets_and_models_dir  --env_name $task_name --which_experiment auditor_train_critic_model --teacher_buffer_save_path ./$datasets_and_models_dir/${env_name_to_expt_dataset_name[$task_name]}/teacher_buffer/baseline/$buffer_name --random_seed 0 --cuda $gpu_id > ./$datasets_and_models_dir/logs/${env_name_to_expt_dataset_name[$task_name]}/auditor/$buffer_name.txt &
    done
done
