import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
import random
from random import sample
from CoLoc_class import CoLoc #import the CoLoc class
import scipy


def get_user_task_dict_by_language(data_path, data_path_save, G, tag_bool, year_list, level, programming_language_std_adjusted, specific_language):
    
    for yr in year_list:
        user_task_count = {}
        user_bool = defaultdict(bool)
        print("task year: ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                ctemp = set([G.nodes[t]['cluster_id'] for t in v[0] if tag_bool[t]])
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))

                if specific_language in ltemp:
                    if not user_bool[v[1]]:
                        user_bool[v[1]] = True
                        user_task_count[v[1]] = defaultdict(float)
                
                    for c in ctemp:
                        user_task_count[v[1]][c] += 1
                        
        fcc_file.close()

        save_obj(user_task_count, f'user_task_count_by_single_year_level_{level}_{yr}_{specific_language}', data_path_save + f'vote_regression_together/user_task_collection/{specific_language}/')

    return


def get_user_starting_year(data_path_save, year_list):
    
    user_starting_year = {}
    
    for yr in year_list:
        print(yr)
        user_task_count = load_obj(f'user_task_count_by_single_year_level_1_{yr}', data_path_save + f'vote_regression_together/user_c_l_list/')
        for u,ud in tqdm(user_task_count.items()):
            if sum(ud.values()) > 0:
                user_starting_year[u] = yr

    save_obj(user_starting_year, f'user_starting_year_dict', data_path_save + f'vote_regression_together/user_task_collection/')

    return

def get_user_starting_year_by_language(data_path_save, year_list, specific_language):
    
    user_starting_year = {}
    for yr in year_list:
        print(yr)
        user_task_count = load_obj(f'user_task_count_by_single_year_level_1_{yr}_{specific_language}', data_path_save + f'vote_regression_together/user_task_collection/{specific_language}/')
        for u,ud in tqdm(user_task_count.items()):
            if sum(ud.values()) > 0:
                user_starting_year[u] = yr

    save_obj(user_starting_year, f'user_starting_year_dict_{specific_language}', data_path_save + f'vote_regression_together/user_task_collection/{specific_language}/')




##! K:[0 tags, 1 user, 2 date, 3 answer]

def build_user_task_dataframe(data_path_save, community_list_core_std, yr_list, yr_label, level, density_user_label, sample_user_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    cc_pmi_matrix_08_12 = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    user_sample_density_bool = load_obj(f'{density_user_label}_bool', data_path_save + f'vote_regression_together/')
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    user_major_language_dict = load_obj(f'user_major_language_{yr_list}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_major_language_dict_all_ties = load_obj(f'user_major_language_all_ties{yr_list}', data_path_save + 'vote_regression_together/user_c_l_list/')

    user_major_language_all_year_dict = load_obj(f'user_major_language_{[yr for yr in range(2008,2024)]}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_major_language_all_year_dict_all_ties = load_obj(f'user_major_language_all_ties{[yr for yr in range(2008,2024)]}', data_path_save + 'vote_regression_together/user_c_l_list/')




    user_list = []
    task_list = []
    year_list = []
    task_count_list = []
    correspond_years_list = []

    density_08_12_list = []
    density_all_year_list = []
    
    major_language_list = []
    major_language_tie_sign_list = []
    major_language_list_all_year = []
    major_language_tie_sign_list_all_year = []
    
    density_user_sign_list = []

    user_task_vector_dict = {}
    user_task_count_dict = {}
    user_bool = defaultdict(bool)
    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if sample_user_bool[u]:
                if not user_bool[u]:
                    user_task_vector_dict[u] = np.zeros(len(community_list_core_std))
                    user_bool[u] = True
                    user_task_count_dict[u] = defaultdict(float)

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_dict[u][C_dict[t]] = 1
                        user_task_count_dict[u][t] += tc

    for u, vb in tqdm(user_task_vector_dict.items()):
        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(vb)).T
        prediction_vector_08_12 = cc_pmi_matrix_08_12.dot(np.array(vb)).T
        for task in community_list_core_std:
            prediction_all_year = prediction_vector_all_year[C_dict[task]]
            prediction_08_12 = prediction_vector_08_12[C_dict[task]]

            user_list.append(u)
            task_list.append(task)
            year_list.append(yr_label)
            correspond_years_list.append(yr_list)

            density_08_12_list.append(prediction_08_12)
            density_all_year_list.append(prediction_all_year)

            major_language_list.append(user_major_language_dict[u])
            major_language_tie_sign_list.append(len(user_major_language_dict_all_ties[u]) if user_major_language_dict_all_ties[u] is not None else None)
            major_language_list_all_year.append(user_major_language_all_year_dict[u])
            major_language_tie_sign_list_all_year.append(len(user_major_language_all_year_dict_all_ties[u]) if user_major_language_all_year_dict_all_ties[u] is not None else None)

            density_user_sign_list.append(1 if user_sample_density_bool[u] else 0)

            task_count_list.append(user_task_count_dict[u][task])

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 'year_label': year_list, 'correspond_years':correspond_years_list, 'task_count': task_count_list, 'density_2008_2012':density_08_12_list, 'density_2008_2023':density_all_year_list, 'major_language': major_language_list, 'major_language_tie_number': major_language_tie_sign_list, 'major_language_all_years': major_language_list_all_year, 'major_language_all_years_tie_number': major_language_tie_sign_list_all_year, 'density_sign': density_user_sign_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/' + f'df_{sample_user_label}_task_with_{density_user_label}_density_{yr_label}_level_{level}.csv')

    return 


def build_user_task_dataframe_with_sample(data_path_save, community_list_core_std, yr_list, yr_label, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    cc_pmi_matrix_08_12 = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    user_sample_density_bool = load_obj(f'{density_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_major_language_dict = load_obj(f'user_major_language_{yr_list}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_major_language_dict_all_ties = load_obj(f'user_major_language_all_ties_{yr_list}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_major_language_all_year_dict = load_obj(f'user_major_language_{[yr for yr in range(2008,2024)]}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_major_language_all_year_dict_all_ties = load_obj(f'user_major_language_all_ties_{[yr for yr in range(2008,2024)]}', data_path_save + 'vote_regression_together/user_c_l_list/')

    user_list = []
    task_list = []
    year_list = []
    task_count_list = []
    correspond_years_list = []

    density_08_12_list = []
    density_all_year_list = []
    
    major_language_list = []
    major_language_tie_sign_list = []
    major_language_list_all_year = []
    major_language_tie_sign_list_all_year = []
    
    density_user_sign_list = []

    user_task_vector_dict = {}
    user_task_count_dict = {}
    user_bool = defaultdict(bool)
    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                if not user_bool[u]:
                    user_task_vector_dict[u] = np.zeros(len(community_list_core_std))
                    user_bool[u] = True
                    user_task_count_dict[u] = defaultdict(float)

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_dict[u][C_dict[t]] = 1
                        user_task_count_dict[u][t] += tc

    for u, vb in tqdm(user_task_vector_dict.items()):
        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(vb)).T
        prediction_vector_08_12 = cc_pmi_matrix_08_12.dot(np.array(vb)).T
        for task in community_list_core_std:
            prediction_all_year = prediction_vector_all_year[C_dict[task]]
            prediction_08_12 = prediction_vector_08_12[C_dict[task]]

            user_list.append(u)
            task_list.append(task)
            year_list.append(yr_label)
            correspond_years_list.append(yr_list)

            density_08_12_list.append(prediction_08_12)
            density_all_year_list.append(prediction_all_year)

            major_language_list.append(user_major_language_dict[u])
            major_language_tie_sign_list.append(len(user_major_language_dict_all_ties[u]) if user_major_language_dict_all_ties[u] is not None else None)
            major_language_list_all_year.append(user_major_language_all_year_dict[u])
            major_language_tie_sign_list_all_year.append(len(user_major_language_all_year_dict_all_ties[u]) if user_major_language_all_year_dict_all_ties[u] is not None else None)

            density_user_sign_list.append(1 if user_sample_density_bool[u] else 0)

            task_count_list.append(user_task_count_dict[u][task])

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 'year_label': year_list, 'correspond_years':correspond_years_list, 'task_count': task_count_list, 'density_2008_2012':density_08_12_list, 'density_2008_2023':density_all_year_list, 'major_language': major_language_list, 'major_language_tie_number': major_language_tie_sign_list, 'major_language_all_years': major_language_list_all_year, 'major_language_all_years_tie_number': major_language_tie_sign_list_all_year, 'density_sign': density_user_sign_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_{sample_percent_label}percent_sample_{density_user_label}_density_{yr_label}_level_{level}.csv')

    return





    
def build_user_activity_dataframe(data_path_save, yr_list, year_label, level, sample_user_label):

    threshold_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    user_major_language_dict = load_obj(f'user_major_language_{yr_list}', data_path_save + 'vote_regression_together/user_c_l_list/')

    user_list = []
    task_list = []
    year_list = []
    task_count_list = []
    major_language_list = []

    user_task_count_dict = {}
    user_bool = defaultdict(bool)
    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if threshold_user_bool[u]:
                if not user_bool[u]:
                    user_bool[u] = True
                    user_task_count_dict[u] = defaultdict(float)

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_count_dict[u][t] += tc
    
    for u, tc_dict in tqdm(user_task_count_dict.items()):
        for t, tc in tc_dict.items(): 
            user_list.append(u)
            task_list.append(t)
            year_list.append(year_label)
            major_language_list.append(user_major_language_dict[u])
            task_count_list.append(user_task_count_dict[u][t])

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 'year_label': year_list, 'task_count': task_count_list, 'major_language': major_language_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_activity_{year_label}_level_{level}.csv')

    return 


    
def build_user_task_dataframe_with_sample_all_language(data_path_save, community_list_core_std, yr_list, yr_label, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    user_list = []
    task_list = []
    year_list = []
    task_count_list = []
    correspond_years_list = []

    density_all_year_list = []
    

    user_task_vector_dict = {}
    user_task_count_dict = {}
    user_bool = defaultdict(bool)
    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                if not user_bool[u]:
                    user_task_vector_dict[u] = np.zeros(len(community_list_core_std))
                    user_bool[u] = True
                    user_task_count_dict[u] = defaultdict(float)

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_dict[u][C_dict[t]] = 1
                        user_task_count_dict[u][t] += tc

    for u, vb in tqdm(user_task_vector_dict.items()):
        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(vb)).T
        for task in community_list_core_std:
            prediction_all_year = prediction_vector_all_year[C_dict[task]]

            user_list.append(u)
            task_list.append(task)
            year_list.append(yr_label)

            density_all_year_list.append(prediction_all_year)

            task_count_list.append(user_task_count_dict[u][task])

            correspond_years_list.append(yr_list)

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 'year_label': year_list, 'correspond_years':correspond_years_list, 'task_count': task_count_list, 'density_2008_2023':density_all_year_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_{sample_percent_label}percent_sample_{density_user_label}_density_{yr_label}_level_{level}_all_language.csv')

    return

def build_user_task_dataframe_with_sample_python(data_path_save, community_list_core_std, yr_list, yr_label, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    user_list = []
    task_list = []
    year_list = []
    task_count_list = []
    correspond_years_list = []

    density_all_year_list = []
    

    user_task_vector_dict = {}
    user_task_count_dict = {}
    user_bool = defaultdict(bool)
    for yr in yr_list:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}_python', data_path_save + 'vote_regression_together/user_task_collection/python/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                if not user_bool[u]:
                    user_task_vector_dict[u] = np.zeros(len(community_list_core_std))
                    user_bool[u] = True
                    user_task_count_dict[u] = defaultdict(float)

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_dict[u][C_dict[t]] = 1
                        user_task_count_dict[u][t] += tc

    for u, vb in tqdm(user_task_vector_dict.items()):
        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(vb)).T
        for task in community_list_core_std:
            prediction_all_year = prediction_vector_all_year[C_dict[task]]

            user_list.append(u)
            task_list.append(task)
            year_list.append(yr_label)
            density_all_year_list.append(prediction_all_year)

            task_count_list.append(user_task_count_dict[u][task])
            correspond_years_list.append(yr_list)

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 'year_label': year_list, 'correspond_years':correspond_years_list, 'task_count': task_count_list, 'density_2008_2023':density_all_year_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_{sample_percent_label}percent_sample_{density_user_label}_density_{yr_label}_level_{level}_python.csv')

    return


##! K:[0 tags, 1 user, 2 date, 3 answer]

def build_user_task_entry_exit_salary_dataframe_with_sample_all_language(data_path_save, community_list_core_std, yr_list1, yr_label1, yr_list2, yr_label2, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    cc_pmi_matrix_08_12 = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    so_yearlist_salary = [2018, 2019, 2020, 2021, 2022, 2023]
    sv_yearlist_salary = [2023]
    topn = 300
    period_label = 'hn_job_task_salary_only_us_log'
    sample_user_label = 'all_threshold_user'

    task_salary_name = [f'{so_yearlist_salary}_{sv_yearlist_salary}_task_salary_topn_{topn}_level_{level}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/']

    task_value_log = load_obj(task_salary_name[0], task_salary_name[1])
    community_list_core_std_no_empty = [c for c in community_list_core_std if task_value_log[c] > 0]
    task_value_vector = [np.exp(task_value_log[t]) for t in community_list_core_std]
    
    

    user_starting_year = load_obj(f'user_starting_year_dict', data_path_save + f'vote_regression_together/user_task_collection/')

    user_task_vector_binary_dict1 = {}
    user_task_count_all_dict1 = {}
    user_task_count_vector_dict1 = {}
    user_bool1 = defaultdict(bool)
    for yr in yr_list1:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                if not user_bool1[u]:
                    
                    user_bool1[u] = True

                    user_task_vector_binary_dict1[u] = np.zeros(len(community_list_core_std))
                    user_task_count_all_dict1[u] = 0
                    user_task_count_vector_dict1[u] = np.zeros(len(community_list_core_std))

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_binary_dict1[u][C_dict[t]] = 1
                        user_task_count_all_dict1[u] += tc
                        user_task_count_vector_dict1[u][C_dict[t]] += tc


    user_task_vector_binary_dict2 = {}
    user_task_count_all_dict2 = {}
    user_bool2 = defaultdict(bool)
    for yr in yr_list2:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                
                if not user_bool2[u]:
                    user_bool2[u] = True
                    user_task_vector_binary_dict2[u] = np.zeros(len(community_list_core_std))
                    user_task_count_all_dict2[u] = 0
                    
                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_binary_dict2[u][C_dict[t]] = 1
                        user_task_count_all_dict2[u] += tc

    user_list = []
    task_list = []
    year_list1 = []
    year_list2 = []
    task_count_list = []
    correspond_years_list1 = []
    correspond_years_list2 = []
    age_list = []
    wage_list = []
    task_value_list = []
    entry_binary_list = []
    task_share_list = []
    user_task_num = []
    all_task_count_list = []
    wage_change_list = []

    density_08_12_list = []
    density_all_year_list = []
    
    user_list12 = list(set([u for u,tc in user_task_count_all_dict1.items() if tc > 0]).intersection(set([u for u,tc in user_task_count_all_dict2.items() if tc > 0])))
    
    for u in tqdm(user_list12):
        uvb = user_task_vector_binary_dict1[u]
        uvb1 = user_task_vector_binary_dict1[u]
        uvb2 = user_task_vector_binary_dict2[u]
        uv = user_task_count_vector_dict1[u]

        share_ut_vector = uv / user_task_count_all_dict1[u]
        age = yr_list1[-1] - user_starting_year[u]
        wage = np.log(share_ut_vector.dot(np.array(task_value_vector)))
        N_u = np.sum(uvb)
        SN_u = user_task_count_all_dict1[u]
        

        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(uvb)).T
        prediction_vector_08_12 = cc_pmi_matrix_08_12.dot(np.array(uvb)).T

        for task in community_list_core_std_no_empty:
            ti = C_dict[task]
            if uvb1[ti] == 0:
                if uvb2[ti] == 1:
                    entry_binary_list.append(1)
                else:
                    entry_binary_list.append(0)
            if uvb1[ti] == 1:
                entry_binary_list.append(2)

            age_list.append(age)
            wage_list.append(wage)
            task_value_list.append(task_value_log[task])
            task_share_list.append(share_ut_vector[ti])
            prediction_all_year = prediction_vector_all_year[ti]
            prediction_08_12 = prediction_vector_08_12[ti]

            user_list.append(u)
            task_list.append(task)
            year_list1.append(yr_label1)
            year_list2.append(yr_label2)
            correspond_years_list1.append(yr_list1)
            correspond_years_list2.append(yr_list2)

            density_08_12_list.append(prediction_08_12)
            density_all_year_list.append(prediction_all_year)

            task_count_list.append(user_task_count_vector_dict1[u][ti])
            user_task_num.append(N_u)
            all_task_count_list.append(SN_u)
            wage_change_list.append(wage - task_value_log[task])

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 
                                'year_label_from': year_list1, 'correspond_years_from':correspond_years_list1,
                                'year_label_to': year_list2, 'correspond_years_to':correspond_years_list2,
                                'task_count': task_count_list, 'density_2008_2012':density_08_12_list,
                                'density_2008_2023':density_all_year_list, "age":age_list, "wage_log":wage_list,
                                'task_share':task_share_list, 'task_value':task_value_list, "entry_sign":entry_binary_list, "user_task_number":user_task_num, 'user_all_task_count':all_task_count_list,
                                'wage_change':wage_change_list})

    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_from_{yr_label1}_to_{yr_label2}_level_{level}_{sample_percent_label}_percent_all_language.csv')

    return 


##! K:[0 tags, 1 user, 2 date, 3 answer]

def build_user_task_entry_exit_salary_dataframe_with_sample_python(data_path_save, community_list_core_std, yr_list1, yr_label1, yr_list2, yr_label2, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label):

    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    cc_pmi_matrix_all_year = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    cc_pmi_matrix_08_12 = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    so_yearlist_salary = [2018, 2019, 2020, 2021, 2022, 2023]
    sv_yearlist_salary = [2023]
    topn = 300
    period_label = 'hn_job_task_salary_only_us_log'

    task_salary_name = [f'{so_yearlist_salary}_{sv_yearlist_salary}_task_salary_topn_{topn}_level_{level}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/']

    task_value_log = load_obj(task_salary_name[0], task_salary_name[1])
    community_list_core_std_no_empty = [c for c in community_list_core_std if task_value_log[c] > 0]
    task_value_vector = [task_value_log[t] for t in community_list_core_std]

    user_starting_year = load_obj(f'user_starting_year_dict_python', data_path_save + f'vote_regression_together/user_task_collection/python/')

    user_task_vector_binary_dict1 = {}
    user_task_count_all_dict1 = {}
    user_task_count_vector_dict1 = {}
    user_bool1 = defaultdict(bool)
    for yr in yr_list1:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}_python', data_path_save + f'vote_regression_together/user_task_collection/python/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                if not user_bool1[u]:
                    
                    user_bool1[u] = True

                    user_task_vector_binary_dict1[u] = np.zeros(len(community_list_core_std))
                    user_task_count_all_dict1[u] = 0
                    user_task_count_vector_dict1[u] = np.zeros(len(community_list_core_std))

                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_binary_dict1[u][C_dict[t]] = 1
                        user_task_count_all_dict1[u] += tc
                        user_task_count_vector_dict1[u][C_dict[t]] += tc


    user_task_vector_binary_dict2 = {}
    user_task_count_all_dict2 = {}
    user_bool2 = defaultdict(bool)
    for yr in yr_list2:
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}_python', data_path_save + f'vote_regression_together/user_task_collection/python/')
        for u, task_dict in tqdm(user_task_count.items()):
            if selected_user_bool[u]:
                
                if not user_bool2[u]:
                    user_bool2[u] = True
                    user_task_vector_binary_dict2[u] = np.zeros(len(community_list_core_std))
                    user_task_count_all_dict2[u] = 0
                    
                for t, tc in task_dict.items():
                    if tc > 0:
                        user_task_vector_binary_dict2[u][C_dict[t]] = 1
                        user_task_count_all_dict2[u] += tc

    user_list = []
    task_list = []
    year_list1 = []
    year_list2 = []
    task_count_list = []
    correspond_years_list1 = []
    correspond_years_list2 = []
    age_list = []
    wage_list = []
    task_value_list = []
    entry_binary_list = []
    task_share_list = []
    user_task_num = []
    all_task_count_list = []
    wage_change_list = []

    density_08_12_list = []
    density_all_year_list = []
    
    user_list12 = list(set([u for u,tc in user_task_count_all_dict1.items() if tc > 0]).intersection(set([u for u,tc in user_task_count_all_dict2.items() if tc > 0])))
    
    for u in tqdm(user_list12):
        uvb = user_task_vector_binary_dict1[u]
        uvb1 = user_task_vector_binary_dict1[u]
        uvb2 = user_task_vector_binary_dict2[u]
        uv = user_task_count_vector_dict1[u]

        share_ut_vector = uv / user_task_count_all_dict1[u]
        age = yr_list1[-1] - user_starting_year[u]
        wage = np.log(share_ut_vector.dot(np.array(task_value_vector)))
        N_u = np.sum(uvb)
        SN_u = user_task_count_all_dict1[u]
        

        prediction_vector_all_year = cc_pmi_matrix_all_year.dot(np.array(uvb)).T
        prediction_vector_08_12 = cc_pmi_matrix_08_12.dot(np.array(uvb)).T

        for task in community_list_core_std_no_empty:
            ti = C_dict[task]

            if uvb1[ti] == 0:
                if uvb2[ti] == 1:
                    entry_binary_list.append(1)
                else:
                    entry_binary_list.append(0)

            if uvb1[ti] == 1:
                entry_binary_list.append(2)

            age_list.append(age)
            wage_list.append(wage)
            task_value_list.append(task_value_log[task])
            task_share_list.append(share_ut_vector[ti])
            prediction_all_year = prediction_vector_all_year[ti]
            prediction_08_12 = prediction_vector_08_12[ti]

            user_list.append(u)
            task_list.append(task)
            year_list1.append(yr_label1)
            year_list2.append(yr_label2)
            correspond_years_list1.append(yr_list1)
            correspond_years_list2.append(yr_list2)

            density_08_12_list.append(prediction_08_12)
            density_all_year_list.append(prediction_all_year)

            task_count_list.append(user_task_count_vector_dict1[u][ti])
            user_task_num.append(N_u)
            all_task_count_list.append(SN_u)
            wage_change_list.append(wage - task_value_log[task])

    df = pd.DataFrame.from_dict({'user_id':user_list, 'task':task_list, 
                                'year_label_from': year_list1, 'correspond_years_from':correspond_years_list1,
                                'year_label_to': year_list2, 'correspond_years_to':correspond_years_list2,
                                'task_count': task_count_list, 'density_2008_2012':density_08_12_list,
                                'density_2008_2023':density_all_year_list, "age":age_list, "wage_log":wage_list,
                                'task_share':task_share_list, 'task_value':task_value_list, "entry_sign":entry_binary_list,"user_task_number":user_task_num, 'user_all_task_count':all_task_count_list,
                                'wage_change':wage_change_list})
    
    print(len(df))

    df.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_from_{yr_label1}_to_{yr_label2}_level_{level}_{sample_percent_label}_percent_python.csv')

    return 

