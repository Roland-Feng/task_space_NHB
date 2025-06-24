import jsonlines
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, date
import numpy as np

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

programming_language_std_adjusted = set(load_obj('programming_language_std_adjusted', data_path_save))

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

experience_period_length = 2

level = 1
print('collect_answer_task_minute_experience_build_dataframe')

G_tag_core = load_obj(f"G_tag_core_with_cut_level{level}", data_path_save + 'networks/probability/')

##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_answer_task_minute_experience_build_dataframe_parallel(yr):

    community_list = list(set([G_tag_core.nodes[n]['cluster_id'] for n in G_tag_core.nodes()]))

    year_list_task = []
    year_list_language = []
    year_list_history = []
    abs_minute_list_task = []
    abs_minute_list_language = []
    abs_minute_list_history = []
    task_list = []
    language_list = []

    task_count_list = []
    language_count_list = []
    history_count_list = []

    task_count_list_log = []
    language_count_list_log = []
    history_count_list_log = []

    ##! 计算user的abs_minute
    user_abs_minute_list_all = defaultdict(list)
    for epi in range(1, experience_period_length+1):
        user_abs_minute_dict_temp =  load_obj(f'user_abs_minute_dict_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, abs_minute_list in user_abs_minute_dict_temp.items():
            user_abs_minute_list_all[u] += abs_minute_list

    user_average_minute_dict = {u:int(np.mean(abs_minute_list)) for u, abs_minute_list in user_abs_minute_list_all.items() if sample_user_bool[u]}
    
    ##! 计算experience
    ##! 统计一些数据
    temp_answer_user = list(user_average_minute_dict.keys())

    user_task_count = {u:defaultdict(float) for u in temp_answer_user}
    user_language_count = {u:defaultdict(float) for u in temp_answer_user}
    user_answer_history = defaultdict(int)

    for epi in range(1, experience_period_length+1):
        user_task_count_temp = load_obj(f'user_task_count_by_single_year_level_{level}_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        user_language_count_temp = load_obj(f'user_language_count_by_single_year_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        user_answer_history_temp = load_obj(f'user_answer_history_single_year_{yr - epi}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u in temp_answer_user:
            if u in user_task_count_temp:
                for task, task_count in user_task_count_temp[u].items():
                    user_task_count[u][task] += task_count
            
            if u in user_language_count_temp:
                for language, language_count in user_language_count_temp[u].items():
                    user_language_count[u][language] += language_count
            
            if u in user_answer_history_temp:
                user_answer_history[u] += user_answer_history_temp[u]

    ##! 计算experience
    ##! 计算 user 的 task 和 language count
    task_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    language_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    answer_history_by_abs_minute = {abs_minute:0 for abs_minute in range(1440)}

    log_task_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    log_language_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    log_answer_history_by_abs_minute = {abs_minute:0 for abs_minute in range(1440)}

    print(yr)
    for abs_minute in tqdm(range(1440)):
        for u in set(temp_answer_user):
            u_absminute_average = user_average_minute_dict[u]
            u_weight = min(1440 - np.abs(u_absminute_average - abs_minute), np.abs(u_absminute_average - abs_minute))
            u_weight = 720 - u_weight
            for task, task_count in user_task_count[u].items():
                task_experience_by_abs_minute[abs_minute][task] += u_weight * task_count
                log_task_experience_by_abs_minute[abs_minute][task] += np.log(u_weight * task_count + 1)
        
            for language, language_count in user_language_count[u].items():
                language_experience_by_abs_minute[abs_minute][language] += u_weight * language_count
                log_language_experience_by_abs_minute[abs_minute][language] += np.log(u_weight * language_count + 1)

            answer_history_by_abs_minute[abs_minute] += u_weight * user_answer_history[u]
            log_answer_history_by_abs_minute[abs_minute] += np.log(u_weight * user_answer_history[u] + 1)


    print("collect data for dataframes")
    for abs_minute in range(1440):
        for task in community_list:
            year_list_task.append(yr)
            task_list.append(task)
            task_count_list.append(task_experience_by_abs_minute[abs_minute][task])
            task_count_list_log.append(log_task_experience_by_abs_minute[abs_minute][task])
            abs_minute_list_task.append(abs_minute)

        for language in programming_language_std_adjusted:
            year_list_language.append(yr)
            language_list.append(language)
            language_count_list.append(language_experience_by_abs_minute[abs_minute][language])
            language_count_list_log.append(log_language_experience_by_abs_minute[abs_minute][language])
            abs_minute_list_language.append(abs_minute)

        year_list_history.append(yr)
        history_count_list.append(answer_history_by_abs_minute[abs_minute])
        history_count_list_log.append(log_answer_history_by_abs_minute[abs_minute])
        abs_minute_list_history.append(abs_minute)

    df_absminute_task = pd.DataFrame.from_dict({'year':year_list_task, 'task': task_list, 'weighted_task_experience':task_count_list, 'weighted_task_experience_log': task_count_list_log,'minute_of_the_day': abs_minute_list_task})
    df_absminute_language = pd.DataFrame.from_dict({'year':year_list_language, 'language': language_list, 'weighted_language_experience':language_count_list, 'weighted_language_experience_log': language_count_list_log, 'minute_of_the_day': abs_minute_list_language})
    df_absminute_history = pd.DataFrame.from_dict({'year':year_list_history, 'weighted_answer_history_experience':history_count_list, 'weighted_answer_history_experience_log':history_count_list_log, 'minute_of_the_day': abs_minute_list_history})

    df_absminute_history.to_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_answer_history_abs_minute_dataframe_{yr}.csv')
    df_absminute_task.to_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_task_abs_minute_dataframe_{yr}.csv')
    df_absminute_language.to_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_language_abs_minute_dataframe_{yr}.csv')

    return