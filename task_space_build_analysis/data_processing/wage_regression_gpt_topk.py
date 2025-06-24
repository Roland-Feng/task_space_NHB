import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict


def bert_skill_embedding_regression_topk(job_skill_dataset, data_path_save, device, data_label):
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device = device)

    occupation_skill_embedding = {}

    for i,skills in tqdm(job_skill_dataset.items()):
        if len(skills) > 0:
            occupation_skill_embedding[i] = model.encode(skills, convert_to_tensor=True)
        
    save_obj(occupation_skill_embedding,f'topk_regression_job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    return


def bert_tag_embedding_regression_topk(community_unweighted_level, data_path_save, device, level, data_label):
    from sentence_transformers import SentenceTransformer

    # Initialize the sentence transformer model
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device = device)

    tag_embedding = {}
    for ns in tqdm(list(community_unweighted_level.values())):
        for n in ns:
            tag_embedding[n] = model.encode(n, convert_to_tensor=True)

    save_obj(tag_embedding, f'topk_regression_tag_embedding_level_{level}', data_path_save + f'jobs/{data_label}/')
    return


def occupation_task_distance_matrix_regression_topk(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, tag_count_dict, tag_count_bool, data_label, top_tags_k):
    
    
    import torch
    from sentence_transformers import util

    tag_community = defaultdict(list)
    for cluster_id, ns in community_unweighted_level.items():
        if len(ns) > 0:
            nstemp = [n for n in ns]
            nctemp = [tag_count_dict[n] for n in nstemp]
            nctemp, nstemp = zip(*sorted(zip(nctemp, nstemp), reverse=True))
            for n in nstemp:
                tag_community[cluster_id].append(n)

    tag_embedding = load_obj(f'topk_regression_tag_embedding_level_{level}', data_path_save + f'jobs/{data_label}/')
    
    if not tag_count_bool:
        community_embedding_average = torch.stack([torch.mean(torch.stack([tag_embedding[t] for t in tag_community[c][:top_tags_k]]), 0) for c in community_list_std if len(community_unweighted_level[c]) > 0])
    else:
        community_tag_count = {c:sum([tag_count_dict[t] for t in tag_community[c]]) for c in community_list_std}
        community_embedding_average = torch.stack([torch.mean(torch.stack([tag_embedding[t] * tag_count_dict[t]/community_tag_count[c][:top_tags_k] for t in tag_community[c]]), 0) for c in community_list_std if len(community_unweighted_level[c]) > 0])
    
    occupation_task_similarity_matrix = {}
    for i,embedding in tqdm(occupation_embedding_dict.items()):
        occupation_task_similarity_matrix[i] = util.cos_sim(embedding, community_embedding_average)

    save_obj(occupation_task_similarity_matrix, f'topk_regression_job_task_similarity_matrix_level_{level}_{top_tags_k}', data_path_save + f'jobs/{data_label}/')

    return



def get_job_community_all_set(occupation_index_set, occupation_task_similarity_matrix, community_list_std_no_empty, skill_similarity_threshold = 0, skill_length_threshold = 2):

    import torch

    job_task_set = {}
    
    for i in occupation_index_set:
        job_task_temp = [community_list_std_no_empty[t] for t, s in zip(torch.max(occupation_task_similarity_matrix[i], dim=1).indices, torch.max(occupation_task_similarity_matrix[i], dim=1).values) if s > skill_similarity_threshold]
        job_task = list(set(job_task_temp))
        if len(job_task) >= skill_length_threshold:
            job_task_set[i] = [t for t in job_task]
        
    return job_task_set



def get_job_task_relatedness(cc_pmi, job_task_dict, C_dict):

    cc_pmi_matrix = np.zeros((len(C_dict), len(C_dict)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    task_relatedness = {}
    for job, tasks in job_task_dict.items():
        temp = []
        for t1 in tasks:
            for t2 in tasks:
                if t1!=t2:
                    temp.append(cc_pmi_matrix[C_dict[t1], C_dict[t2]])

        if len(temp) > 0:
            task_relatedness[job] = sum(temp) / len(temp)

    return task_relatedness


def get_job_task_salary_average(community_salary_by_sv_year, job_task_dict, occupation_index_set, salary_type = 'mean'):
    job_average_task_salary = {}

    if salary_type == 'mean':
        for o in occupation_index_set:
            job_average_task_salary[o] = np.mean([community_salary_by_sv_year[c] for c in job_task_dict[o]])

    if salary_type == 'max':
        for o in occupation_index_set:
            job_average_task_salary[o] = np.max([community_salary_by_sv_year[c] for c in job_task_dict[o]])

    return job_average_task_salary


def get_task_ubiquity_in_so_user(data_path_save, so_year_list, level):

    task_ubiquity_in_so = defaultdict(int)
    user_bool = defaultdict(bool)
    for yr in so_year_list:
        task_user_set = load_obj(f'task_user_set_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, uts in tqdm(task_user_set.items()):
            for c in uts:
                if not user_bool[u]:
                    task_ubiquity_in_so[c] += 1
                    user_bool[u] = True

    return task_ubiquity_in_so


def get_job_task_ubiquity_average(job_task_dict, occupation_index_set, task_ubiquity_so_user):
    job_task_ubi_average = {}
    for o in occupation_index_set:
        job_task_ubi_average[o] = np.mean([task_ubiquity_so_user[t] for t in job_task_dict[o]])

    return job_task_ubi_average



def build_regression_dataframe_topk(data_path_save,level,community_list_std,community_unweighted_level,skill_similarity_threshold,skill_length_threshold,so_year_list, task_salary_name, data_label, density_user_label, top_tags_k, salary_type = 'mean'):

    C_dict = {c:i for i,c in enumerate(community_list_std)}
    tag_community_dict = {}
    for c, tags in community_unweighted_level.items():
        for t in tags:
            tag_community_dict[t] = c
    
    ##! salary
    print("get salary")
    salary_dict = load_obj(f'regression_job_salary_dict', data_path_save + f'jobs/{data_label}/')
    occupation_index_set = set(salary_dict.keys())

    ##! occupation task length
    print('get task length')
    occupation_task_similarity_matrix = load_obj(f'topk_regression_job_task_similarity_matrix_level_{level}_{top_tags_k}', data_path_save + f'jobs/{data_label}/')
    community_list_std_no_empty = [c for c in community_list_std if len(community_unweighted_level[c]) > 0]

    occupation_index_set = occupation_index_set.intersection(occupation_task_similarity_matrix.keys())
    job_task_dict = get_job_community_all_set(occupation_index_set, occupation_task_similarity_matrix, community_list_std_no_empty, skill_similarity_threshold, skill_length_threshold)
    occupation_index_set = occupation_index_set.intersection(job_task_dict.keys())


    ##! relatedness
    print('get relatedness of tasks')
    cc_pmi = load_obj(f'cc_pmi_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    job_task_relatedness_average = get_job_task_relatedness(cc_pmi, job_task_dict, C_dict)
    occupation_index_set = occupation_index_set.intersection(job_task_relatedness_average.keys())

    ##! community salary
    print('get task salary')
    community_salary_by_sv_year = load_obj(task_salary_name[0],task_salary_name[1])
    job_average_task_salary = get_job_task_salary_average(community_salary_by_sv_year, job_task_dict, occupation_index_set, salary_type)

    ##! task ubiquity
    print('get task ubiquity')
    task_ubiquity_so_user = get_task_ubiquity_in_so_user(data_path_save, so_year_list, level)
    job_task_ubiquity_average = get_job_task_ubiquity_average(job_task_dict, occupation_index_set, task_ubiquity_so_user)

    ##! job year
    job_year_regression_dict = load_obj(f'regression_job_year_dict', data_path_save + f'jobs/{data_label}/')

    
    occupation_index = list(occupation_index_set)
    salary_list = [salary_dict[oi] for oi in occupation_index]
    task_length_list = [len(job_task_dict[oi]) for oi in occupation_index]
    task_relatedness_list = [job_task_relatedness_average[oi] for oi in occupation_index]
    job_task_average_salary_list = [job_average_task_salary[oi] for oi in occupation_index]
    job_task_ubiquity_average_list = [job_task_ubiquity_average[oi] for oi in occupation_index]
    job_year_list = [job_year_regression_dict[oi] for oi in occupation_index]

    print('build dataframe')
    df = pd.DataFrame.from_dict({'occupation_index': occupation_index, 
                                'salary': salary_list, 
                                'job_task_length': task_length_list,
                                'job_task_relatedness': task_relatedness_list,
                                'job_task_salary_average': job_task_average_salary_list,
                                'job_task_ubiquity_average': job_task_ubiquity_average_list,
                                'job_year': job_year_list})
    
    return df

