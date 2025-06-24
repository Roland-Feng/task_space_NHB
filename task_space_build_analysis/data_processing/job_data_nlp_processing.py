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



def bert_skill_embedding(job_skill_dataset, data_path_save, device, data_label):
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device = device)

    occupation_skill_embedding = {}

    for i,skills in tqdm(job_skill_dataset.items()):
        if len(skills) > 0:
            occupation_skill_embedding[i] = model.encode(skills, convert_to_tensor=True)
        
    save_obj(occupation_skill_embedding,f'job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    return


def bert_tag_embedding(community_unweighted_level, data_path_save, device, level, data_label):
    from sentence_transformers import SentenceTransformer

    # Initialize the sentence transformer model
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device = device)

    tag_embedding = {}
    for ns in tqdm(list(community_unweighted_level.values())):
        for n in ns:
            tag_embedding[n] = model.encode(n, convert_to_tensor=True)

    save_obj(tag_embedding, f'tag_embedding_level_{level}', data_path_save + f'jobs/{data_label}/')
    return


def occupation_task_distance_matrix(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, tag_count_dict, tag_count_bool, data_label):
    
    
    import torch
    from sentence_transformers import util

    tag_community = defaultdict(list)
    for cluster_id, ns in community_unweighted_level.items():
        for n in ns:
            tag_community[cluster_id].append(n)

    tag_embedding = load_obj(f'tag_embedding_level_{level}', data_path_save + f'jobs/{data_label}/')
    
    if not tag_count_bool:
        community_embedding_average = torch.stack([torch.mean(torch.stack([tag_embedding[t] for t in tag_community[c]]), 0) for c in community_list_std if len(community_unweighted_level[c]) > 0])
    else:
        community_tag_count = {c:sum([tag_count_dict[t] for t in tag_community[c]]) for c in community_list_std}
        community_embedding_average = torch.stack([torch.mean(torch.stack([tag_embedding[t] * tag_count_dict[t]/community_tag_count[c] for t in tag_community[c]]), 0) for c in community_list_std if len(community_unweighted_level[c]) > 0])
    
    occupation_task_similarity_matrix = {}
    for i,embedding in tqdm(occupation_embedding_dict.items()):
        occupation_task_similarity_matrix[i] = util.cos_sim(embedding, community_embedding_average)

    save_obj(occupation_task_similarity_matrix, f'job_task_similarity_matrix_level_{level}', data_path_save + f'jobs/{data_label}/')

    return



##! build user set
def get_job_community_set(occupation_task_similarity_matrix ,community_list_std, community_list_std_no_empty, skill_similarity_threshold = 0):

    C_dict = {c:i for i,c in enumerate(community_list_std)}

    from random import shuffle, seed
    import torch

    seed(83)

    job_task_training_vector = {}
    job_task_training_set = {}
    job_task_target_set = {}

    for i, similarity_matrix in tqdm(occupation_task_similarity_matrix.items()):
        job_task_temp = [community_list_std_no_empty[t] for t, s in zip(torch.max(similarity_matrix, dim=1).indices, torch.max(similarity_matrix, dim=1).values) if s > skill_similarity_threshold]
        job_task = list(set(job_task_temp))
        shuffle(job_task)
        ##! job至少有多少task #################################################################################
        ##? job至少有多少task #################################################################################
        ##* job至少有多少task #################################################################################
        if len(job_task) >= 3:
            target_sign = int(len(job_task) * 0.4)
            target_task = job_task[:target_sign]
            training_task = job_task[target_sign:]

            job_task_training_set[i] = [t for t in training_task]
            job_task_target_set[i] = [t for t in target_task]
            job_task_training_vector[i] = [0 for c in community_list_std]
            for t in training_task:
                job_task_training_vector[i][C_dict[t]] = 1

    return job_task_training_vector, job_task_training_set, job_task_target_set


##! make prediction - ignore target
def predict_user_community_ignore_target(cc_pmi_matrix, user_community_vector_binary, community_list_std, user_target):
    C_dict = {c:i for i,c in enumerate(community_list_std)}
    user_prediction = {}
    for u,vb in tqdm(user_community_vector_binary.items()):
        cc_copy = cc_pmi_matrix.copy()
        for c in user_target[u]:
            cc_copy[:,C_dict[c]] = 0

        for i in range(len(community_list_std)):
            if cc_copy[i,:].sum() != 0:
                cc_copy[i,:] = cc_copy[i,:] / cc_copy[i,:].sum()
    
        user_prediction[u] = cc_copy.dot(np.array(vb)).T

    return user_prediction


##! examine prediction
def examine_prediction(user_prediction, user_target, user_source, community_list_std):
    user_list = set(user_target.keys()).intersection(set(user_source.keys()))
    community_prediction = defaultdict(list)
    for u in tqdm(user_list):
        prediction = user_prediction[u]
        for i,p in enumerate(prediction):
            c = community_list_std[i]
            if c not in user_source[u]:
                if c in user_target[u]:
                    community_prediction[c].append((p, 1))
                if c not in user_target[u]:
                    community_prediction[c].append((p, 0))

    community_prediction_all = []
    for cp in list(community_prediction.values()):
        community_prediction_all += cp

    return community_prediction, community_prediction_all



