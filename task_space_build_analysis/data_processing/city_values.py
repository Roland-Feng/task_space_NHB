import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
import numpy as np
import jsonlines
from scipy.sparse import csr_matrix, hstack, vstack
from collections import Counter, defaultdict


def get_efua_task_values(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    efua_users_dict = load_obj(f'efua_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_efua_dict = load_obj(f'user_efua_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')

    user_bool = defaultdict(bool)
    for u in user_efua_dict.keys():
        user_bool[u] = True

    user_task_total = defaultdict(int)
    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                user_task_total[u] += sum(utcdict.values())


    efua_task_vector = {efua:np.zeros(len(community_list_core_std)) for efua in efua_users_dict.keys()}
    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    efua_task_vector[user_efua_dict[u]][C_dict[t]] += tc / user_task_total[u] if user_task_total[u] > 0 else 0

    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    efua_tv = {efua:efuatask / np.sum(efuatask) if np.sum(efuatask) > 0 else np.zeros(len(community_list_core_std)) for efua, efuatask in tqdm(efua_task_vector.items())}
    efua_values = {efua:efuatask@task_salary_vector for efua, efuatask in tqdm(efua_tv.items())}
    
    return efua_values


def get_country_task_values(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    country_users_dict = load_obj(f'country_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_country_dict = load_obj(f'user_country_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')

    user_bool = defaultdict(bool)
    for u in user_country_dict.keys():
        user_bool[u] = True

    user_task_total = defaultdict(int)
    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                user_task_total[u] += sum(utcdict.values())

    country_task_vector = {country:np.zeros(len(community_list_core_std)) for country in country_users_dict.keys()}
    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    country_task_vector[user_country_dict[u]][C_dict[t]] += tc / user_task_total[u] if user_task_total[u] > 0 else 0


    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    country_tv = {country:countrytask / np.sum(countrytask) if np.sum(countrytask) > 0 else np.zeros(len(community_list_core_std)) for country, countrytask in tqdm(country_task_vector.items())}

    country_values = {country:countrytask@task_salary_vector for country, countrytask in tqdm(country_tv.items())}
    
    return country_values



def get_efua_task_values_quantile(data_path_save, task_salary_name, year_period, sample_user_label, level, top_percent):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    efua_users_dict = load_obj(f'efua_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_efua_dict = load_obj(f'user_efua_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')

    user_bool = defaultdict(bool)
    for u in user_efua_dict.keys():
        user_bool[u] = True

    user_task_vector = {u:np.zeros(len(community_list_core_std)) for u, ub in user_bool.items() if ub}

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    user_task_vector[u][C_dict[t]] += tc

    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    efua_salary_dict = {}
    for efua, user_list in (efua_users_dict.items()):
        temp = [user_task_vector[u]@task_salary_vector/np.sum(user_task_vector[u]) for u in user_list if np.sum(user_task_vector[u]) > 0]
        if len(temp) > 0:
            efua_salary_dict[efua] = np.quantile(temp, top_percent)
    
    return efua_salary_dict


def get_country_task_values_quantile(data_path_save, task_salary_name, year_period, sample_user_label, level, top_percent):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    
    country_users_dict = load_obj(f'country_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_country_dict = load_obj(f'user_country_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')

    user_bool = defaultdict(bool)
    for u in user_country_dict.keys():
        user_bool[u] = True

    user_task_vector = {u:np.zeros(len(community_list_core_std)) for u, ub in user_bool.items() if ub}

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    user_task_vector[u][C_dict[t]] += tc

    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    country_salary_dict = {}
    for country, user_list in (country_users_dict.items()):
        temp = [user_task_vector[u]@task_salary_vector/np.sum(user_task_vector[u]) for u in user_list if np.sum(user_task_vector[u]) > 0]
        if len(temp) > 0:
            country_salary_dict[country] = np.quantile(temp, top_percent)
    
    return country_salary_dict





def confine_users_geo_data(data_path_save, user_label, year_period_list):
    
    user_bool = load_obj(f'{user_label}_bool', data_path_save + f'vote_regression_together/')

    for year_period in year_period_list:
        print(year_period)
        efua_users_dict = load_obj(f'efua_users_dict_{year_period}',data_path_save + 'surveys/user_geo/')
        country_users_dict = load_obj(f'country_users_dict_{year_period}',data_path_save + 'surveys/user_geo/')
        user_efua_dict = load_obj(f'user_efua_dict_{year_period}',data_path_save + 'surveys/user_geo/')
        user_country_dict = load_obj(f'user_country_dict_{year_period}',data_path_save + 'surveys/user_geo/')

        efua_users_dict_temp = {efua:[u for u in ulist if user_bool[u]] for efua, ulist in efua_users_dict.items()}
        save_obj(efua_users_dict_temp, f'efua_users_dict_{year_period}_{user_label}', data_path_save + 'surveys/user_geo/')
        print(user_label, len(efua_users_dict_temp), 'efua length')
        del efua_users_dict

        country_users_dict_temp = {country:[u for u in ulist if user_bool[u]] for country, ulist in country_users_dict.items()}
        save_obj(country_users_dict_temp, f'country_users_dict_{year_period}_{user_label}', data_path_save + 'surveys/user_geo/')
        print(user_label, len(country_users_dict_temp), 'country length')
        del country_users_dict

        user_efua_dict_temp = {u:efua for u, efua in user_efua_dict.items() if user_bool[u]}
        save_obj(user_efua_dict_temp, f'user_efua_dict_{year_period}_{user_label}', data_path_save + 'surveys/user_geo/')
        print(user_label, len(user_efua_dict_temp), 'user with efua length')
        del user_efua_dict

        user_country_dict_temp = {u:country for u, country in user_country_dict.items() if user_bool[u]}
        save_obj(user_country_dict_temp, f'user_country_dict_{year_period}_{user_label}', data_path_save + 'surveys/user_geo/')
        print(user_label, len(user_country_dict_temp), 'user with country length')
        del user_country_dict

    return

def collect_city_task_matrix(data_path_save, year_list, level, sample_user_label, user_efua_dict):
    
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_bool = defaultdict(bool)
    efua_count_dict = defaultdict(int)
    for u, efua in user_efua_dict.items():
        if sample_user_bool[u]:
            user_bool[u] = True
            efua_count_dict[efua] += 1

    efua_count_tuple = sorted(efua_count_dict.items(), key = lambda kv:(-kv[1], kv[0]))
    efua_list_std = [ec[0] for ec in efua_count_tuple]
    EFUA_dict = {c:i for i,c in enumerate(efua_list_std)}

    efua_task_matrix = np.zeros((len(efua_list_std),len(community_list_core_std)))

    for yr in tqdm(year_list):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    efua_task_matrix[EFUA_dict[user_efua_dict[u]], C_dict[t]] += tc

    save_obj(efua_task_matrix, f'efua_task_matrix_{year_list}_level_{level}_{sample_user_label}', data_path_save + 'surveys/user_geo/')
    save_obj(efua_list_std, f'efua_list_std_{year_list}_level_{level}_{sample_user_label}', data_path_save + 'surveys/user_geo/')


def city_task_rca(data_path_save, year_list, level, sample_user_label):
    
    efua_task_matrix = load_obj(f'efua_task_matrix_{year_list}_level_{level}_{sample_user_label}', data_path_save + 'surveys/user_geo/')

    cooccur_matrix = efua_task_matrix
    
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
    
    rca_matrix = np.where(rca_matrix > 1, rca_matrix, 0)

    save_obj(rca_matrix, f'efua_task_matrix_rca_{year_list}_level_{level}_{sample_user_label}', data_path_save + 'surveys/user_geo/')




def get_efua_task_values_by_aggregating_tasks(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}

    efua_users_dict = load_obj(f'efua_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_efua_dict = load_obj(f'user_efua_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_bool = defaultdict(bool)
    efua_count_dict = defaultdict(int)
    for u, efua in user_efua_dict.items():
        if sample_user_bool[u]:
            user_bool[u] = True
            efua_count_dict[efua] += 1

    efua_count_tuple = sorted(efua_count_dict.items(), key = lambda kv:(-kv[1], kv[0]))
    efua_list_std = [ec[0] for ec in efua_count_tuple]
    EFUA_dict = {c:i for i,c in enumerate(efua_list_std)}

    efua_task_matrix = np.zeros((len(efua_list_std),len(community_list_core_std)))

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    efua_task_matrix[EFUA_dict[user_efua_dict[u]], C_dict[t]] += tc


    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    efua_values = {efua:efua_task_matrix[efua_i]@task_salary_vector / np.sum(efua_task_matrix[efua_i]) for efua, efua_i in EFUA_dict.items() if np.sum(efua_task_matrix[efua_i]) > 0}
    
    return efua_values


def get_country_task_values_by_aggregating_tasks(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}

    country_users_dict = load_obj(f'country_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_country_dict = load_obj(f'user_country_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_bool = defaultdict(bool)
    country_count_dict = defaultdict(int)
    for u, country in user_country_dict.items():
        if sample_user_bool[u]:
            user_bool[u] = True
            country_count_dict[country] += 1

    country_count_tuple = sorted(country_count_dict.items(), key = lambda kv:(-kv[1], kv[0]))
    country_list_std = [ec[0] for ec in country_count_tuple]
    country_dict = {c:i for i,c in enumerate(country_list_std)}

    country_task_matrix = np.zeros((len(country_list_std),len(community_list_core_std)))

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    country_task_matrix[country_dict[user_country_dict[u]], C_dict[t]] += tc


    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    country_values = {country:country_task_matrix[country_i]@task_salary_vector / np.sum(country_task_matrix[country_i]) for country, country_i in country_dict.items() if np.sum(country_task_matrix[country_i]) > 0}
    
    return country_values




def get_efua_task_values_by_aggregating_tasks_rca(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}

    efua_users_dict = load_obj(f'efua_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_efua_dict = load_obj(f'user_efua_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_bool = defaultdict(bool)
    efua_count_dict = defaultdict(int)
    for u, efua in user_efua_dict.items():
        if sample_user_bool[u]:
            user_bool[u] = True
            efua_count_dict[efua] += 1

    efua_count_tuple = sorted(efua_count_dict.items(), key = lambda kv:(-kv[1], kv[0]))
    efua_list_std = [ec[0] for ec in efua_count_tuple]
    EFUA_dict = {c:i for i,c in enumerate(efua_list_std)}

    efua_task_matrix = np.zeros((len(efua_list_std),len(community_list_core_std)))

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    efua_task_matrix[EFUA_dict[user_efua_dict[u]], C_dict[t]] += tc


    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    cooccur_matrix = efua_task_matrix
    
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
    
    rca_matrix = np.where(rca_matrix > 1, rca_matrix, 0)
            

    efua_values = {efua:rca_matrix[efua_i]@task_salary_vector / np.sum(rca_matrix[efua_i]) for efua, efua_i in EFUA_dict.items() if np.sum(rca_matrix[efua_i]) > 0}
    
    return efua_values


def get_country_task_values_by_aggregating_tasks_rca(data_path_save, task_salary_name, year_period, sample_user_label, level):

    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_core_std)}

    country_users_dict = load_obj(f'country_users_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    user_country_dict = load_obj(f'user_country_dict_{year_period}_{sample_user_label}',data_path_save + 'surveys/user_geo/')
    
    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    user_bool = defaultdict(bool)
    country_count_dict = defaultdict(int)
    for u, country in user_country_dict.items():
        if sample_user_bool[u]:
            user_bool[u] = True
            country_count_dict[country] += 1

    country_count_tuple = sorted(country_count_dict.items(), key = lambda kv:(-kv[1], kv[0]))
    country_list_std = [ec[0] for ec in country_count_tuple]
    country_dict = {c:i for i,c in enumerate(country_list_std)}

    country_task_matrix = np.zeros((len(country_list_std),len(community_list_core_std)))

    for yr in tqdm(year_period):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utcdict in user_task_count.items():
            if user_bool[u]:
                for t, tc in utcdict.items():
                    country_task_matrix[country_dict[user_country_dict[u]], C_dict[t]] += tc


    task_salary = load_obj(task_salary_name[0], task_salary_name[1])
    task_salary_vector = np.zeros(len(community_list_core_std))
    for t, ts in task_salary.items():
        task_salary_vector[C_dict[t]] = ts

    cooccur_matrix = country_task_matrix
    
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
    
    rca_matrix = np.where(rca_matrix > 1, rca_matrix, 0)
            

    country_values = {country:rca_matrix[country_i]@task_salary_vector / np.sum(rca_matrix[country_i]) for country, country_i in country_dict.items() if np.sum(rca_matrix[country_i]) > 0}
    
    return country_values