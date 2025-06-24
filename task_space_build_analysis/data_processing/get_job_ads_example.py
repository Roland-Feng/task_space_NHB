from job_data_nlp_processing import *

data_path = '/home/xiangnan/task_space_code/task_space_data/'
#data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {t[0]:t[1] for t in tag_count}

device = 'cpu'
job_skill_dataset = load_obj(f'job_skill_dataset', data_path_save + f'jobs/{data_label}/')
level = 1

print('level:  ', level)

tag_count_bool = False

community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
tag_community_dict = {}
for i,c in community_unweighted_level.items():
    for t in c:
        tag_community_dict[t] = i

occupation_embedding_dict = load_obj(f'job_skill_embedding', data_path_save + f'jobs/{data_label}/')

skill_similarity_threshold = 0.1

occupation_task_similarity_matrix = load_obj(f'job_task_similarity_matrix_level_{level}', data_path_save + f'jobs/{data_label}/')

##! build training and target
print('build training and target')
community_list_std_no_empty = [c for c in community_list_std if len(community_unweighted_level[c]) > 0]

import torch
i = 0
job_ads = {}
with jsonlines.open(f'{data_path}hnhiring_jobs.json', 'r') as fcc_file:
    for job_sample in fcc_file:
        for ji, js in tqdm(enumerate(job_sample)):
            job_ads[str(ji)] = js


job_matched_tasks_dict = {}
for i, similarity_matrix in tqdm(occupation_task_similarity_matrix.items()):
    job_task_matched = [(community_list_std_no_empty[t],s) for t, s in zip(torch.max(similarity_matrix, dim=1).indices, torch.max(similarity_matrix, dim=1).values)]
    job_matched_tasks_dict[i] = job_task_matched

save_obj(job_matched_tasks_dict, 'job_matched_tasks_dict', data_path_save + f'jobs/{data_label}/')

    