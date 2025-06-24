import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
from random import sample
from CoLoc_class import CoLoc #import the CoLoc class
import scipy


##! frequency check
#region - frequency check
def check_language_frequence(data_path, data_path_save, programming_language_std_adjusted):

    freq_in_year_question = {yr:{l:0 for l in programming_language_std_adjusted} for yr in range(2008,2024)}
    
    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))
                for l in ltemp:
                    freq_in_year_question[yr][l] += 1
        fcc_file.close()

    save_obj(freq_in_year_question, 'freq_in_year_question', data_path_save + 'vote_regression_together/user_c_l_list/')


    freq_in_year_answer = {yr:{l:0 for l in programming_language_std_adjusted} for yr in range(2008,2024)}

    for yr in range(2008,2024):
        print("answer:  ", yr)
        with jsonlines.open(f'{data_path_save}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))
                for l in ltemp:
                    freq_in_year_answer[yr][l] += 1
        fcc_file.close()

    save_obj(freq_in_year_answer, 'freq_in_year_answer', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


##! [0: parent_answer_num, 1: parent_answer_vote, 2: parent_vote, 3: parent_id]
def vote_distribution_check(data_path_save):
    question_vote = {yr:{} for yr in range(2008, 2024)}
    question_answer_vote = {yr:{} for yr in range(2008, 2024)}
    for yr in tqdm(range(2008,2024)):
        answer_parent_vote = load_obj(f'answer_parent_num_vote_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        question_vote[yr] = {answer_parent_vote[k][3]:answer_parent_vote[k][2] for k in answer_parent_vote.keys()}
        question_answer_vote[yr] = {answer_parent_vote[k][3]:answer_parent_vote[k][2] for k in answer_parent_vote.keys()}

    save_obj(question_vote,'question_vote', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(question_answer_vote,'question_answer_vote', data_path_save + 'vote_regression_together/user_c_l_list/')

    return

def user_task_frequency_check(data_path_save, community_list_core_std, yr_list, level):
    
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    user_task_vector_dict = {}
    user_task_count_dict = {}
    user_bool = defaultdict(bool)

    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if not user_bool[u]:
                user_task_vector_dict[u] = np.zeros(len(community_list_core_std))
                user_bool[u] = True
                user_task_count_dict[u] = defaultdict(float)

            for t, tc in task_dict.items():
                if tc > 0:
                    user_task_vector_dict[u][C_dict[t]] = 1
                    user_task_count_dict[u][t] += tc

    user_task_count_freq = {}
    user_task_count_binary = {}
    for u, vb in tqdm(user_task_vector_dict.items()):
        user_task_count_binary[u] = np.sum(vb)
        user_task_count_freq[u] = sum([tc for tc in user_task_count_dict[u].values()])

    return user_task_count_freq, user_task_count_binary


def user_answer_number_check(data_path_save, yr_list):
    
    user_answer_number_dict = defaultdict(int)
    
    for yr in yr_list:
        user_answer_history = load_obj(f'user_answer_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, uah in tqdm(user_answer_history.items()):
            user_answer_number_dict[u] += uah

    
    return user_answer_history


def user_answer_number_check_with_threshold(data_path_save, threshold, year_period):
    
    user_answer_number_dict = defaultdict(int)
    
    for yr in tqdm(range(2008, 2024)):
        user_answer_history = load_obj(f'user_answer_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, uah in user_answer_history.items():
            user_answer_number_dict[u] += uah

    user_bool = defaultdict(bool)
    for u, uah in user_answer_number_dict.items():
        if uah >= threshold:
            user_bool[u] = True

    period_uah_list = []

    period_uah_len = []
    for yr in tqdm(range(2008 + year_period + 1, 2024)):
        uah_temp = defaultdict(float)
        yr_list = [yr - year_period + i for i in range(year_period)]
        for yrtemp in yr_list:
            user_answer_history = load_obj(f'user_answer_history_single_year_{yrtemp}',  data_path_save + 'vote_regression_together/user_c_l_list/')
            for u, uah in user_answer_history.items():
                if user_bool[u]:
                    uah_temp[u] += uah

        period_uah_list += [uah for uah in uah_temp.values()]
        period_uah_len.append(len(uah_temp))

    print(period_uah_len)

    return period_uah_list

#endregion
