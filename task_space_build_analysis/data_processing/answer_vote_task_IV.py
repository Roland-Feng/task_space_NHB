import jsonlines
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, date
import numpy as np


def get_answer_user_dict(data_path, data_path_save):

    answer_user_dict = {}

    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                answer_user_dict[k] = v[1]

        fcc_file.close()

    save_obj(answer_user_dict, f'answer_user_dict_all', data_path_save + 'vote_regression_together/df_task_IV/')

def get_answer_vote_rank_confine_user(data_path, data_path_save, sample_user_label):

    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    answer_upvote_count = load_obj('answer_upvote_count', data_path)
    answer_rank_dict = defaultdict(int)
    answer_bool = defaultdict(bool)

    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if sample_user_bool[v[1]]:
                    answer_bool[k] = True

        fcc_file.close()

    import scipy.stats as scipystats

    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if len(v[3]) > 0:
                    temp_answer_id = []
                    temp_answer_vote = []
                    for a in v[3]:
                        if answer_bool[a]:
                            temp_answer_id.append(a)
                            if answer_upvote_count.get(a) is not None:
                                temp_answer_vote.append(-answer_upvote_count[a])
                            else:
                                temp_answer_vote.append(0)

                    if len(temp_answer_vote) > 0:
                        r = (scipystats.rankdata(temp_answer_vote, 'dense')).astype(int)

                        for ir,ans_id in zip(r, temp_answer_id):
                            answer_rank_dict[ans_id] = int(ir)
                
        fcc_file.close()


    save_obj(answer_rank_dict, f'answer_rank_dict_{sample_user_label}', data_path_save + 'vote_regression_together/df_task_IV/')



def get_yr_task_abs_minute_exp(data_path_save, level, experience_period_length, yr, sample_user_label):

    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    ##! 计算user的abs_minute
    print("calculate abs minute of users")
    user_abs_minute_list_all = defaultdict(list)
    for epi in range(1, experience_period_length+1):
        user_abs_minute_dict_temp =  load_obj(f'user_abs_minute_dict_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, abs_minute_list in user_abs_minute_dict_temp.items():
            user_abs_minute_list_all[u] += abs_minute_list

    user_average_minute_dict = {u:int(np.mean(abs_minute_list)) for u, abs_minute_list in user_abs_minute_list_all.items() if sample_user_bool[u]}
    
    ##! 计算experience
    ##! 统计一些数据
    print("collect experience")
    temp_answer_user = list(user_average_minute_dict.keys())
    
    user_task_count = {u:defaultdict(float) for u in temp_answer_user}
    
    for epi in range(1, experience_period_length+1):
        user_task_count_temp = load_obj(f'user_task_count_by_single_year_level_{level}_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u in temp_answer_user:
            if u in user_task_count_temp:
                for task, task_count in user_task_count_temp[u].items():
                    user_task_count[u][task] += task_count

    ##! 计算experience
    ##! 计算 user 的 task 和 language count
    print("calculate weighted experience")
    log_task_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}

    for abs_minute in tqdm(range(1440)):

        for u in set(temp_answer_user):
            u_absminute_average = user_average_minute_dict[u]
            u_weight = min(1440 - np.abs(u_absminute_average - abs_minute), np.abs(u_absminute_average - abs_minute))
            u_weight = 720 - u_weight
            for task, task_count in user_task_count[u].items():
                log_task_experience_by_abs_minute[abs_minute][task] += np.log(u_weight * task_count + 1)

    save_obj(log_task_experience_by_abs_minute, f'task_abs_minute_exp_log_{sample_user_label}_{yr}_period_{experience_period_length}_level_{level}', data_path_save+'vote_regression_together/df_task_IV/')

    return


def get_yr_history_abs_minute_exp(data_path_save, experience_period_length, yr, sample_user_label):

    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    
    print(yr)
    
    ##! 计算user的abs_minute
    print("calculate abs minute of users")
    user_abs_minute_list_all = defaultdict(list)
    for epi in range(1, experience_period_length+1):
        user_abs_minute_dict_temp =  load_obj(f'user_abs_minute_dict_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, abs_minute_list in user_abs_minute_dict_temp.items():
            user_abs_minute_list_all[u] += abs_minute_list

    user_average_minute_dict = {u:int(np.mean(abs_minute_list)) for u, abs_minute_list in user_abs_minute_list_all.items() if sample_user_bool[u]}
    
    ##! 计算experience
    ##! 统计一些数据
    print("collect experience")
    temp_answer_user = list(user_average_minute_dict.keys())
    
    user_answer_history = defaultdict(int)

    for epi in range(1, experience_period_length+1):
        user_answer_history_temp = load_obj(f'user_answer_history_single_year_{yr - epi}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u in temp_answer_user:
            if u in user_answer_history_temp:
                user_answer_history[u] += user_answer_history_temp[u]

    ##! 计算experience
    ##! 计算 user 的 task 和 language count
    print("calculate weighted experience")

    log_answer_history_by_abs_minute = {abs_minute:0 for abs_minute in range(1440)}

    for abs_minute in tqdm(range(1440)):
        for u in set(temp_answer_user):
            u_absminute_average = user_average_minute_dict[u]
            u_weight = min(1440 - np.abs(u_absminute_average - abs_minute), np.abs(u_absminute_average - abs_minute))
            u_weight = 720 - u_weight

            log_answer_history_by_abs_minute[abs_minute] += np.log(u_weight * user_answer_history[u] + 1)


    save_obj(log_answer_history_by_abs_minute, f'history_abs_minute_log_{sample_user_label}_{yr}_period_{experience_period_length}', data_path_save+'vote_regression_together/df_task_IV/')

    return


def vote_task_regression_IV_json(data_path_save, experience_period_length, level, sample_user_label, yr):
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    answer_rank_dict = load_obj(f'answer_rank_dict_{sample_user_label}', data_path_save + 'vote_regression_together/df_task_IV/')
    answer_rank_dict_all = load_obj(f'answer_rank_dict_all_answer_user', data_path_save + 'vote_regression_together/df_task_IV/')


    with jsonlines.open(f'{data_path_save}vote_regression_together/answer_vote_coefficients_json/vote_task_exp_IV_{sample_user_label}_level_{level}_single_year_{yr}_period_{experience_period_length}.json', 'w') as w:
        with jsonlines.open(f'{data_path_save}vote_regression_together/answer_vote_coefficients_json/answer_vote_coefficients_level_{level}_single_year_{yr}_period_{experience_period_length}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                u_id = v[14]
                if sample_user_bool[u_id]:
                    answer_rank = v[23]
                    parent_ans_number = v[2]
                    a_absminute = v[33]
                    q_absminute = v[34]
                    answer_day_from_zero = v[21]
                    question_day_from_zero = v[39]
                    aq_minute = (answer_day_from_zero - question_day_from_zero) * 1440 + a_absminute - q_absminute
                    task_id = v[10]
                    task_experience = v[6]
                    general_experience = v[8]
                    answer_time_rank = v[31]
                    answer_rank_threshold_user = answer_rank_dict[k]
                    answer_vote = v[9]
                    all_answer_vote = v[3]
                    answer_rank_all_users = answer_rank_dict_all[k]


                    ##! 0. answer_rank, 1. parent_ans_number, 2. aq_minute 3. a_absminute, 4. q_absminute
                    ##! 5. task_id, 6. task_experience, 7. general_experience, 8. answer_time_rank
                    ##! 9. answer_rank_threshold_users, 10. answer_vote, 11. vote of answer under parent
                    ##! 12. answer_day_from_zero, 13. question_day_from_zero, 14. answer_rank_all_users

                    w.write({k:[answer_rank, parent_ans_number, aq_minute, a_absminute, q_absminute,
                                task_id, task_experience, general_experience, answer_time_rank,
                                answer_rank_threshold_user, answer_vote, all_answer_vote,
                                answer_day_from_zero, question_day_from_zero, answer_rank_all_users]})
                    
        w.close()
    fcc_file.close()

    return




def vote_task_exp_IV_build_dataframe(data_path_save, experience_period_length, level, sample_user_label, yr_list):

    data = []
    data_general_experience = []
    answer_user_dict = load_obj(f'answer_user_dict_all', data_path_save + 'vote_regression_together/df_task_IV/')

    for yr in yr_list:
        ##! log_task_experience_by_abs_minute[abs_minute][task]
        log_t_exp = load_obj(f'task_abs_minute_exp_log_{sample_user_label}_{yr}_period_{experience_period_length}_level_{level}', data_path_save+'vote_regression_together/df_task_IV/')

        log_history = load_obj(f'history_abs_minute_log_{sample_user_label}_{yr}_period_{experience_period_length}', data_path_save+'vote_regression_together/df_task_IV/')


        ##! 0. answer_rank, 1. parent_ans_number, 2. aq_minute 3. a_absminute, 4. q_absminute
        ##! 5. task_id, 6. task_experience, 7. general_experience, 8. answer_time_rank
        ##! 9. answer_rank_threshold_user, 10. answer_vote, 11. vote of answer under parent
        ##! 12. answer_day_from_zero, 13. question_day_from_zero, 14. answer_rank_all_users

        answer_bool = defaultdict(bool)

        with jsonlines.open(f'{data_path_save}vote_regression_together/answer_vote_coefficients_json/vote_task_exp_IV_{sample_user_label}_level_{level}_single_year_{yr}_period_{experience_period_length}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                top_threshold = 1 if v[9] == 1 else 0
                top_all = 1 if v[14] == 1 else 0
                if v[8] == 1:
                    data.append([k, log_t_exp[v[4]][v[5]], top_threshold, v[1], v[6], 
                                v[5], yr, v[2], v[3], v[4],
                                v[7], v[10], v[11], 
                                v[5]+'_'+str(v[4]), v[5]+'_'+str(yr), log_history[v[4]],
                                v[12],v[13], top_all, 
                                answer_user_dict[k], str(yr) + '_' + answer_user_dict[k]])
                    
                    if not answer_bool[k]:
                        answer_bool[k] = True
                        data_general_experience.append([k, log_t_exp[v[4]][v[5]], top_threshold, v[1], v[6], 
                                                        v[5], yr, v[2], v[3], v[4],
                                                        v[7], v[10], v[11], 
                                                        v[5]+'_'+str(v[4]), v[5]+'_'+str(yr), log_history[v[4]],
                                                        v[12], v[13], top_all, 
                                                        answer_user_dict[k], str(yr) + '_' + answer_user_dict[k]])

        fcc_file.close()

        del log_t_exp

    df = pd.DataFrame(data, 
                    columns=['answer_id', 'task_IV_log', 'answer_top', 'parent_ans_number', 'task_experience',
                            'task_id', 'year','aq_minute', 'a_absminute', 'q_absminute',
                            'general_experience', 'answer_vote', 'all_answer_vote', 
                            'group_task_qminute', 'group_task_year', 'history_IV_log',
                            'a_abs_day', 'q_abs_day', 'answer_top_all_users',
                            'user_id', 'user_year'])
    

    df.to_csv(f'{data_path_save}vote_regression_together/df_task_IV/df_vote_task_exp_IV_{sample_user_label}_level_{level}_all_year_period_{experience_period_length}.csv')
    

    df_general = pd.DataFrame(data_general_experience, 
                    columns=['answer_id', 'task_IV_log', 'answer_top', 'parent_ans_number', 'task_experience',
                            'task_id', 'year','aq_minute', 'a_absminute', 'q_absminute',
                            'general_experience', 'answer_vote', 'all_answer_vote', 
                            'group_task_qminute', 'group_task_year', 'history_IV_log',
                            'a_abs_day', 'q_abs_day', 'answer_top_all_users',
                            'user_id', 'user_year'])
    
    df_general.to_csv(f'{data_path_save}vote_regression_together/df_task_IV/df_vote_general_exp_IV_{sample_user_label}_level_{level}_all_year_period_{experience_period_length}.csv')
    return