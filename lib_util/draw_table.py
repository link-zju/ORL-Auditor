import os
import pandas as pd
import argparse
import glob
import json
import copy
import numpy as np

def unroll(data):
    if isinstance(data, dict):
        for key, value in data.items():
            # Recursively unroll the next level and prepend the key to each row.
            for row in unroll(value):
                print(row)
                yield [key] + row
    if not isinstance(data, dict):
        # This is the bottom of the structure (defines exactly one row).
        yield data



def draw_table(env_name_list, suspect_model_type_list, metric_list, json_data_dir, num_shadow_student, significance_level, trajectory_size):
    # check the needed files
    # json_file_list = glob.glob(f'{json_data_dir}/*/*numstu_{num_shadow_student}-*trajsize_{trajectory_size}-*signlevel_{significance_level}*')
    # assert json_file_list == len(env_name_list) * len(suspect_model_type_list)
    
    results = {}
    for env_name in env_name_list:
        results[env_name] = {}
        for suspect_model_type in suspect_model_type_list:
            results[env_name][suspect_model_type] = {}
            json_file_path_list = glob.glob(f'{json_data_dir}/*/*envname_{env_name}-*sustype_{suspect_model_type}-*numstu_{num_shadow_student}-*trajsize_{trajectory_size}-*signlevel_{significance_level}-*.json', recursive=True)
            assert 1 == len(json_file_path_list), print(json_file_path_list)
            json_file_path = json_file_path_list[0]
            with open(json_file_path, 'r') as j:
                # contents = json.loads(j.read())
                json_results = json.loads(j.read())
            
            json_results_copy = copy.deepcopy(json_results)
            json_results_copy = json_results_copy["results_for_auditing_dataset"]
            for metric in metric_list:
                TPR_list = []
                TNR_list = []
                for audit_dataset_name in json_results_copy.keys():
                    TPR_list.append(json_results_copy[audit_dataset_name][metric]["True Positive Rate"])
                    TNR_list.append(json_results_copy[audit_dataset_name][metric]["True Negative Rate"])

                TPR_arr = np.array(TPR_list)
                TNR_arr = np.array(TNR_list)
                
                results[env_name][suspect_model_type][metric] = {"TPR": [f"{TPR_arr.mean()*100:0.2f}$\pm${TPR_arr.std()*100:0.2f}"], "TNR": [f"{TNR_arr.mean()*100:0.2f}$\pm${TNR_arr.std()*100:0.2f}"]}

    df = pd.DataFrame(list(unroll(results)))
    # print(df.columns)
    df.columns = ["task_name", "suspect_model_type", "distance", "TXR", "value"]
    df = df.sort_values(by=['distance', 'task_name'], ascending=[True, False])
    
    
    # df.reindex(labels=metric_list)
    df = pd.pivot(df, index = ["task_name", "suspect_model_type"], columns=["distance", "TXR"], values="value")
    # df = df.sort_values(by=["task_name", "suspect_model_type"], ascending=[False, True])
    # df = df.sort_values(by=["task_name"], ascending=[False])
    # df = pd.json_normalize(results, max_level=0)
    df.to_excel(os.path.join(json_data_dir, f"audit_results-numstu_{num_shadow_student}-trajsize_{trajectory_size}-signlevel_{significance_level}.xlsx"))
    
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    
    parser.add_argument("--env_name_list", type=str, default="LunarLanderContinuous-v2 BipedalWalker-v3 Ant-v2")
    parser.add_argument("--suspect_model_type_list", type=str, default="BC BCQ IQL TD3PlusBC")
    
    parser.add_argument("--json_data_dir", type=str, default="")
    parser.add_argument("--num_shadow_student", type=int, default=15)
    parser.add_argument("--significance_level", type=float, default=0.01)
    parser.add_argument("--trajectory_size", type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    
    metric_list = ["l1_distance", "l2_distance", "cos_distance", "wasserstein_distance"]
    
    env_name_list = list(args.env_name_list.split("\n"))
    suspect_model_type_list = list(args.suspect_model_type_list.split("\n"))
    
    draw_table(env_name_list, suspect_model_type_list, metric_list, args.json_data_dir, args.num_shadow_student, args.significance_level, args.trajectory_size)
    