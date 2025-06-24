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


##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_task_language_matrix_answer(data_path, data_path_save, yr_list, level, programming_language_std_adjusted, community_list_core_std, tag_bool, tag_community_dict, user_bool):
    community_dict = {c:i for i, c in enumerate(community_list_core_std)}
    language_dict = {l:i for i, l in enumerate(programming_language_std_adjusted)}
    tl_matrix = np.zeros((len(community_list_core_std), len(programming_language_std_adjusted)))
    language_set = set(programming_language_std_adjusted)

    for yr in yr_list:
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if user_bool[v[1]]:
                    ltemp = language_set.intersection(set(v[0]))
                    ctemp = set([tag_community_dict[t] for t in v[0] if tag_bool[t]])
                    for l in ltemp:
                        for c in ctemp:
                            tl_matrix[community_dict[c], language_dict[l]] += 1

        fcc_file.close()

    save_obj(tl_matrix, f'answer_task_language_matrix_{yr_list}_level_{level}', data_path_save + f'task_language_nestedness/tl_matrix/')


##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_task_language_matrix_question(data_path, data_path_save, yr_list, level, programming_language_std_adjusted, community_list_core_std, tag_bool, tag_community_dict, user_bool):
    community_dict = {c:i for i, c in enumerate(community_list_core_std)}
    language_dict = {l:i for i, l in enumerate(programming_language_std_adjusted)}
    tl_matrix = np.zeros((len(community_list_core_std), len(programming_language_std_adjusted)))
    language_set = set(programming_language_std_adjusted)


    for yr in yr_list:
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if user_bool[v[1]]:
                    ltemp = language_set.intersection(set(v[0]))
                    ctemp = set([tag_community_dict[t] for t in v[0] if tag_bool[t]])
                    for l in ltemp:
                        for c in ctemp:
                            tl_matrix[community_dict[c], language_dict[l]] += 1

        fcc_file.close()

    save_obj(tl_matrix, f'question_task_language_matrix_{yr_list}_level_{level}', data_path_save + f'task_language_nestedness/tl_matrix/')


def calculate_task_language_pmi_sig(tl_matrix, language_list, community_list, p_value):

    df = pd.DataFrame(data=tl_matrix, columns=language_list, index=community_list)

    Q = CoLoc(df)

    df_Q = Q.make_sigPMIpci(p_value)

    return df_Q[df_Q > 0].fillna(0)


def calculate_task_language_rca(tl_matrix):
    cooccur_matrix = tl_matrix
    
    sum1 = np.sum(cooccur_matrix, axis = 1)
    sum2 = np.sum(cooccur_matrix, axis = 0)
    sum0 = np.sum(cooccur_matrix)
    
    rca_matrix = np.zeros(cooccur_matrix.shape)
    
    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[1]):
            if sum1[i] != 0 and sum2[j] != 0:
                rca_matrix[i][j] = (cooccur_matrix[i][j]/sum0) / (sum1[i]/sum0 * sum2[j]/sum0)
            else:
                rca_matrix[i][j] = 0

    return rca_matrix



##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_task_language_user_count_matrix_answer(data_path_save, yr_list, level, programming_language_std_adjusted, community_list_core_std, user_bool):
    community_dict = {c:i for i, c in enumerate(community_list_core_std)}
    language_dict = {l:i for i, l in enumerate(programming_language_std_adjusted)}
    tl_uc_matrix = np.zeros((len(community_list_core_std), len(programming_language_std_adjusted)))

    tl_user_set = defaultdict(set)

    for yr in yr_list:
        user_tl_count = load_obj(f'user_task_language_count_by_single_year_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, tl_count in user_tl_count.items():
            if user_bool[u]:
                for tl in tl_count.keys():
                    tl_user_set[tl].add(u)

    for tl, uset in tl_user_set.items():
        tl_uc_matrix[community_dict[tl[0]], language_dict[tl[1]]] = len(uset)

    save_obj(tl_uc_matrix, f'answer_task_language_user_count_matrix_{yr_list}_level_{level}', data_path_save + f'task_language_nestedness/tl_matrix/')

##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_task_language_user_count_matrix_answer_with_merge(data_path_save, yr_list, level, programming_language_std_adjusted, community_list_core_std, user_bool, language_merge_dict):
    language_merge_temp = {l:l for l in programming_language_std_adjusted}
    for l, lm in language_merge_dict.items():
        for lt in lm:
            language_merge_temp[lt] = l

    community_dict = {c:i for i, c in enumerate(community_list_core_std)}
    language_dict = {l:i for i, l in enumerate(programming_language_std_adjusted)}
    tl_uc_matrix = np.zeros((len(community_list_core_std), len(programming_language_std_adjusted)))

    tl_user_set = defaultdict(set)

    for yr in yr_list:
        user_tl_count = load_obj(f'user_task_language_count_by_single_year_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, tl_count in user_tl_count.items():
            if user_bool[u]:
                for tl in tl_count.keys():
                    tl_user_set[(tl[0], language_merge_temp[tl[1]])].add(u)

    for tl, uset in tl_user_set.items():
        tl_uc_matrix[community_dict[tl[0]], language_dict[tl[1]]] = len(uset)

    save_obj(tl_uc_matrix, f'answer_task_language_user_count_matrix_with_merge_{yr_list}_level_{level}', data_path_save + f'task_language_nestedness/tl_matrix/')



def collect_task_language_user_count_matrix_answer_with_merge_all_year(data_path_save, yr_list, level, programming_language_std_adjusted, community_list_core_std, user_bool, language_merge_dict):
    language_merge_temp = {l:l for l in programming_language_std_adjusted}
    for l, lm in language_merge_dict.items():
        for lt in lm:
            language_merge_temp[lt] = l

    community_dict = {c:i for i, c in enumerate(community_list_core_std)}
    language_dict = {l:i for i, l in enumerate(programming_language_std_adjusted)}
    tl_uc_matrix = np.zeros((len(community_list_core_std), len(programming_language_std_adjusted)))

    tl_user_set = defaultdict(set)

    for yr in yr_list:
        user_tl_count = load_obj(f'user_task_language_count_by_single_year_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, tl_count in user_tl_count.items():
            if user_bool[u]:
                for tl in tl_count.keys():
                    tl_user_set[(tl[0], language_merge_temp[tl[1]])].add(u)

    for tl, uset in tl_user_set.items():
        tl_uc_matrix[community_dict[tl[0]], language_dict[tl[1]]] = len(uset)

    save_obj(tl_uc_matrix, f'tl_user_count_matrix_all_year_level_{level}', data_path_save + f'task_language_nestedness/tl_matrix/')


##! K:[0 tags, 1 user, 2 date, 3 answer]
def collect_language_user_count_matrix_answer_with_merge(data_path_save, yr_list, programming_language_std_adjusted, user_bool, language_merge_dict):
    language_merge_temp = {l:l for l in programming_language_std_adjusted}
    for l, lm in language_merge_dict.items():
        for lt in lm:
            language_merge_temp[lt] = l

    l_user_set = defaultdict(set)

    for yr in yr_list:
        user_l_count = load_obj(f'user_language_count_by_single_year_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, l_count in user_l_count.items():
            if user_bool[u]:
                for l in l_count.keys():
                    l_user_set[language_merge_temp[l]].add(u)
    
    l_uc_count = {l:len(uset) for l, uset in l_user_set.items()}

    save_obj(l_uc_count, f'answer_language_user_count_matrix_with_merge_{yr_list}', data_path_save + f'task_language_nestedness/tl_matrix/')



def collect_language_user_count_matrix_answer_with_merge_all_year(data_path_save, yr_list, programming_language_std_adjusted, user_bool, language_merge_dict):
    language_merge_temp = {l:l for l in programming_language_std_adjusted}
    for l, lm in language_merge_dict.items():
        for lt in lm:
            language_merge_temp[lt] = l

    l_user_set = defaultdict(set)

    for yr in yr_list:
        user_l_count = load_obj(f'user_language_count_by_single_year_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, l_count in user_l_count.items():
            if user_bool[u]:
                for l in l_count.keys():
                    l_user_set[language_merge_temp[l]].add(u)
    
    l_uc_count = {l:len(uset) for l, uset in l_user_set.items()}

    save_obj(l_uc_count, f'language_user_count_matrix_all_year_level', data_path_save + f'task_language_nestedness/tl_matrix/')
