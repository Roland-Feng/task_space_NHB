import importlib
import job_data_nlp_processing_threshold
importlib.reload(job_data_nlp_processing_threshold)
from job_data_nlp_processing_threshold import *

#data_path = '/home/xiangnan/task_space_code/task_space_data/'
data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {t[0]:t[1] for t in tag_count}

device = 1
job_skill_dataset = load_obj(f'job_skill_dataset', data_path_save + f'jobs/{data_label}/')
bert_skill_embedding_threshold(job_skill_dataset, data_path_save, device, data_label)

for level in range(1,2):
    print('level:  ', level)
    tag_count_bool = False

    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    tag_community_dict = {}
    for i,c in community_unweighted_level.items():
        for t in c:
            tag_community_dict[t] = i

    bert_tag_embedding_threshold(community_unweighted_level, data_path_save, device, level, data_label)
    
    occupation_embedding_dict = load_obj(f'threshold_job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    occupation_task_distance_matrix_threshold(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, tag_count_dict, tag_count_bool, data_label)


import importlib
import job_data_nlp_processing_threshold
importlib.reload(job_data_nlp_processing_threshold)
from job_data_nlp_processing_threshold import *


#data_path = '/home/xiangnan/task_space_code/task_space_data/'
data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

density_user_label = 'all_threshold_user'

skill_similarity_threshold = 0.4

for level in range(1,2):
    print('level:  ', level)
    tag_count_bool = False

    ##! load data
    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    tag_community_dict = {}
    for i,c in community_unweighted_level.items():
        for t in c:
            tag_community_dict[t] = i

    cc_pmi_matrix = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    occupation_task_similarity_matrix = load_obj(f'threshold_job_task_similarity_matrix_level_{level}', data_path_save + f'jobs/{data_label}/')

    ##! build training and target
    print('build training and target')
    community_list_std_no_empty = [c for c in community_list_std if len(community_unweighted_level[c]) > 0]
    job_task_training_vector, job_task_training_set, job_task_target_set = get_job_community_set_threshold(occupation_task_similarity_matrix ,community_list_std, community_list_std_no_empty, skill_similarity_threshold)

    #job_task_prediction = predict_user_community(cc_pmi_matrix, job_task_training_vector)
    job_task_prediction = predict_user_community_ignore_target_threshold(cc_pmi_matrix, job_task_training_vector, community_list_std, job_task_target_set)
    community_prediction, cp_all = examine_prediction_threshold(job_task_prediction, job_task_target_set, job_task_training_set, community_list_std)

    save_obj(cp_all, f'threshold_cp_all_{level}_4', data_path_save + f'jobs/{data_label}/')


import importlib
import wage_regression_gpt_threshold #import the module here, so that it can be reloaded.
importlib.reload(wage_regression_gpt_threshold)
from wage_regression_gpt_threshold import *

#data_path = '/home/xiangnan/task_space_code/task_space_data/'
data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {t[0]:t[1] for t in tag_count}

device = 1
job_skill_dataset = load_obj(f'regression_job_skill_dict', data_path_save + f'jobs/{data_label}/')
bert_skill_embedding_regression_threshold(job_skill_dataset, data_path_save, device, data_label)

for level in range(1,2):
    print('level:  ', level)
    tag_count_bool = False

    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    tag_community_dict = {}
    for i,c in community_unweighted_level.items():
        for t in c:
            tag_community_dict[t] = i

    bert_tag_embedding_regression_threshold(community_unweighted_level, data_path_save, device, level, data_label)
    
    occupation_embedding_dict = load_obj(f'threshold_regression_job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    occupation_task_distance_matrix_regression_threshold(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, tag_count_dict, tag_count_bool, data_label)




import importlib
import wage_regression_gpt_threshold #import the module here, so that it can be reloaded.
importlib.reload(wage_regression_gpt_threshold)
from wage_regression_gpt_threshold import *

#data_path = '/home/xiangnan/task_space_code/task_space_data/'
data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

skill_similarity_threshold = 0.4
skill_length_threshold = 1

data_label = 'HN_data_gpt35'
density_user_label = 'all_threshold_user'

##! salary year: SO 2018 to 2022, SV: 2019

so_yearlist = [2018, 2019, 2020,2021,2022,2023]


so_yearlist_salary = [2018, 2019, 2020, 2021, 2022, 2023]
sv_yearlist_salary = [2023]
period_label = 'hn_job_task_salary_only_us_log'
topn = 300

salary_type = 'mean'

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

job_task_dict = {}

tag_count_bool = False

for level in range(1,2):
    tag_bool_core = load_obj(f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

    task_salary_name = [f'{so_yearlist_salary}_{sv_yearlist_salary}_task_salary_topn_{topn}_level_{level}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/']

    df = build_regression_dataframe_threshold(data_path_save, level, community_list_std, community_unweighted_level, skill_similarity_threshold,skill_length_threshold,so_yearlist, task_salary_name, data_label, density_user_label, salary_type)
    df.to_csv(data_path_save + f'jobs/{data_label}/threshold_{period_label}_{so_yearlist_salary}_{sv_yearlist_salary}_df_regression_level_{level}_topn_{topn}_{sample_user_label}_4.csv')
    