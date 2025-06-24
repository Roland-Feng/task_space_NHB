import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize


def get_efua_task_values(data_path_save, task_salary_name, year_period, sample_user_label, level, pls_str):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    pls_user_all = defaultdict(set)
    pls_task_vector = {}
    for yr in tqdm(year_period):
        pls_users_dict = load_obj(f'{pls_str}_users_{yr}',data_path_save + 'user_location/')
        for pls, userlist in pls_users_dict.items():
            if pls not in pls_task_vector:
                pls_task_vector[pls] = np.zeros(len(community_list_core_std))

            pls_user_all[pls] = pls_user_all[pls] | set(userlist)
            


    user_task_count_all = {}
    user_task_count_bool = defaultdict(bool)
    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utc in user_task_count.items():
            if user_bool[u]:
                if not user_task_count_bool[u]:
                    user_task_count_bool[u] = True
                    user_task_count_all[u] = defaultdict(int)

                for t, tc in utc.items():
                    user_task_count_all[u][t] += tc



    for pls, userset in pls_user_all.items():
        for u in userset:
            if user_bool[u] and user_task_count_bool[u]:
                for t, tc in user_task_count_all[u].items():
                    pls_task_vector[pls][C_dict[t]] += tc


    pls_list_std = [pls for pls in pls_task_vector.keys()]
    cooccur_matrix = np.array([pls_task_vector[pls] for pls in pls_list_std])
    
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
    
    rca_matrix = np.where(rca_matrix > 1, 1, 0)
    rca_matrix_normalized = normalize(rca_matrix, 'l1')

    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_list = [task_salary[t] for t in community_list_core_std]

    pls_value_vector = rca_matrix_normalized@task_salary_list
    pls_values = {pls:pls_value_vector[i] for i, pls in enumerate(pls_list_std) if pls_value_vector[i] != 0}

    return pls_values, {pls:len(userlist) for pls, userlist in pls_user_all.items()}