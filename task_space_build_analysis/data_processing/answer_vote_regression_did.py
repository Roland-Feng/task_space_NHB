import jsonlines
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, date
import numpy as np


##! K:[0 tags, 1 user, 2 date, 3 answer]

##? collect dataset
#region - collect dataset

def get_user_task_dict(data_path, data_path_save, G, tag_bool, year_list, level):
    
    for yr in year_list:
        user_task_count = {}
        user_bool = defaultdict(bool)
        print("task year: ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                if not user_bool[v[1]]:
                    user_bool[v[1]] = True
                    user_task_count[v[1]] = defaultdict(float)

                ctemp = set([G.nodes[t]['cluster_id'] for t in v[0] if tag_bool[t]])

                if len(ctemp) > 0:
                    for c in ctemp:
                        user_task_count[v[1]][c] += 1
                        
        fcc_file.close()

        save_obj(user_task_count, f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


def get_user_language_dict(data_path, data_path_save, programming_language_std_adjusted, year_list):
    
    for yr in year_list:
        user_language_count = {}
        user_bool = defaultdict(bool)
        print("language year: ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                if not user_bool[v[1]]:
                    user_bool[v[1]] = True
                    user_language_count[v[1]] = defaultdict(float)
                    
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))

                if len(ltemp) > 0:
                    for l in ltemp:
                        user_language_count[v[1]][l] += 1
                        
        fcc_file.close()

        save_obj(user_language_count, f'user_language_count_by_single_year_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


def get_user_task_language_dict(data_path, data_path_save, programming_language_std_adjusted, year_list, G_tag_core, tag_bool_core, level):
    for yr in year_list:
        user_tl_count = {}
        user_bool = defaultdict(bool)
        print("task year: ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                if not user_bool[v[1]]:
                    user_bool[v[1]] = True
                    user_tl_count[v[1]] = defaultdict(float)

                ctemp = set([G_tag_core.nodes[t]['cluster_id'] for t in v[0] if tag_bool_core[t]])
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))

                if len(ctemp) > 0 and len(ltemp) > 0:
                    for c in ctemp:
                        for l in ltemp:
                            user_tl_count[v[1]][(c,l)] += 1
                        
        fcc_file.close()

        save_obj(user_tl_count, f'user_task_language_count_by_single_year_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')


def get_user_answer_tagall_set(data_path, data_path_save):
    for yr in range(2008,2024):
        tags_user_list = defaultdict(list)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                tags_user_list[v[1]] += v[0]

            tag_user_set = {k:set(v) for k,v in tags_user_list.items()}
        fcc_file.close()
        
        save_obj(tag_user_set,f'tag_user_set_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return

def get_answer_task_set(data_path_save, tag_bool_core, year_list, level):

    for yr in tqdm(year_list):
        tag_user_set = load_obj(f'tag_user_set_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        community_core_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
        tag_community_dict = {t:c for c, ts in community_core_level.items() for t in ts}
        task_user_set = {u:set([tag_community_dict[t] for t in tagset if tag_bool_core[t]]) for u, tagset in tag_user_set.items()}
    
        save_obj(task_user_set,f'task_user_set_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


##! [0: parent_answer_num, 1: parent_answer_vote, 2: parent_vote, 3: parent_id]
def get_answer_parent_vote(data_path, data_path_save):

    question_upvote_count = load_obj('question_upvote_count', data_path)
    answer_upvote_count = load_obj('answer_upvote_count', data_path)

    question_answer_num = {}
    question_answer_vote = {}
    question_vote = {}

    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                question_answer_num[k] = len(v[3])
                temp = 0
                
                for a in v[3]:
                    if answer_upvote_count.get(a) is not None:
                        temp += answer_upvote_count[a]
                    
                question_answer_vote[k] = temp
                question_vote[k] = question_upvote_count[k] if k in question_upvote_count else 0
        fcc_file.close()

    ##! [0: parent_answer_num, 1: parent_answer_vote, 2: parent_vote, 3: parent_id]
    for yr in range(2008,2024):
        print("answer:  ", yr)
        answer_parent_vote = {}
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                p = v[3]
                answer_parent_vote[k] = [question_answer_num[p], question_answer_vote[p], question_vote[p], p]

        fcc_file.close()
        save_obj(answer_parent_vote, f'answer_parent_num_vote_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


##! answer rank
def get_answer_vote_rank(data_path, data_path_save):

    answer_upvote_count = load_obj('answer_upvote_count', data_path)
    answer_rank_dict = defaultdict(int)

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
                        temp_answer_id.append(a)
                        if answer_upvote_count.get(a) is not None:
                            temp_answer_vote.append(answer_upvote_count[a])
                        else:
                            temp_answer_vote.append(0)
                    
                    temp_answer_vote_sorted, temp_answer_id_sorted = zip(*sorted(zip(temp_answer_vote, temp_answer_id), reverse=True))
                    for i,ans_id in enumerate(temp_answer_id_sorted):
                        answer_rank_dict[ans_id] = i + 1
                
        fcc_file.close()
        save_obj(answer_rank_dict, f'answer_rank_dict_alltime', data_path_save + 'vote_regression_together/user_c_l_list/')


    ##! user's top-1 answer count
    for yr in range(2008,2024):
        user_top_1_answer_count = defaultdict(int)
        print("answer:  ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                u_id = v[1]
                if answer_rank_dict[u_id] == 1:
                    user_top_1_answer_count[u_id] += 1

        fcc_file.close()
        save_obj(user_top_1_answer_count, f'user_top_1_answer_count_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    return


##! answer hour
def get_question_answer_time(data_path, data_path_save):

    question_hour_dict = defaultdict(int)
    answer_hour_dict = defaultdict(int)

    question_abs_minute_dict = defaultdict(int)
    answer_abs_minute_dict = defaultdict(int)

    
    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                question_hour_dict[k] = int(v[2][11:13])
                question_abs_minute_dict[k] = int(v[2][11:13]) * 60 + int(v[2][14:16])


        fcc_file.close()
    save_obj(question_hour_dict, 'question_hour_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(question_abs_minute_dict, 'question_abs_minute_dict', data_path_save + 'vote_regression_together/user_c_l_list/')

    ##! answer hours
    user_hour_dict_global = defaultdict(list)
    for yr in range(2008,2024):
        user_hour_dict = defaultdict(list)
        user_abs_minute_dict = defaultdict(list)

        print("answer:  ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                answer_hour_dict[k] = int(v[2][11:13])
                user_hour_dict[v[1]].append(int(v[2][11:13]))
                user_hour_dict_global[v[1]].append(int(v[2][11:13]))
                user_abs_minute_dict[v[1]].append(int(v[2][11:13]) * 60 + int(v[2][14:16]))
                answer_abs_minute_dict[k] = int(v[2][11:13]) * 60 + int(v[2][14:16])


        fcc_file.close()
        save_obj(user_hour_dict, f'user_hour_dict_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        save_obj(user_abs_minute_dict, f'user_abs_minute_dict_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

    save_obj(answer_hour_dict, 'answer_hour_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(answer_abs_minute_dict, 'answer_abs_minute_dict', data_path_save + 'vote_regression_together/user_c_l_list/')

    average_user_hour_global = defaultdict(int)
    median_user_hour_global = defaultdict(int)

    for u, hl in user_hour_dict_global.items():
        average_user_hour_global[u] = np.mean(hl)
        median_user_hour_global[u] = np.median(hl)

    save_obj(average_user_hour_global, 'average_user_answer_hour_all_time', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(median_user_hour_global, f'median_user_answer_hour_all_time',  data_path_save + 'vote_regression_together/user_c_l_list/')
    
    return


##! answer hour
def get_answer_time_rank(data_path, data_path_save):
    answer_time_dict = defaultdict(int)

    ##! answer hours
    for yr in range(2008,2024):
        
        print("answer:  ", yr)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                answer_time_dict[k] = v[2]

        fcc_file.close()
    
    save_obj(answer_time_dict, f'answer_time_dict', data_path_save + 'vote_regression_together/user_c_l_list/')

    answer_time_rank = defaultdict(int)
    for yr in range(2008,2024):
        print("question:  ", yr)
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                tempa = [a for a in v[3]]
                tempt = [answer_time_dict[a] for a in v[3]]
                if len(tempa) > 0:
                    tempt, tempa = zip(*sorted(zip(tempt,tempa), reverse=False))
                    for i, a in enumerate(tempa):
                        answer_time_rank[a] = i + 1
                    
        fcc_file.close()

    save_obj(answer_time_rank, 'answer_time_rank', data_path_save + 'vote_regression_together/user_c_l_list/')
    
    return


def get_user_answer_history(data_path, data_path_save):
    for yr in range(2008,2024):
        user_answer_history = defaultdict(int)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                user_answer_history[v[1]] += 1

        fcc_file.close()
        save_obj(user_answer_history, f'user_answer_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
    return 


def get_answer_task_language_length(data_path, data_path_save, programming_language_std_adjusted, G, tag_bool, level):
    
    for yr in range(2008,2024):
        answer_language_length = defaultdict(list)
        answer_task_length = defaultdict(list)
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                ltemp = programming_language_std_adjusted.intersection(set(v[0]))
                ctemp = set([G.nodes[t]['cluster_id'] for t in v[0] if tag_bool[t]])
                answer_language_length[len(ltemp)].append(k)
                answer_task_length[len(ctemp)].append(k)

        fcc_file.close()
    
        save_obj(answer_language_length, f'answer_language_length_{level}_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        save_obj(answer_task_length, f'answer_task_length_{level}_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')

    return 

def get_python_user(data_path, data_path_save, top_user_post_count, python_threshold = 0.1):

    programming_language_std_adjusted = set(load_obj('programming_language_std_adjusted', data_path_save))
    
    user_python_count = {}
    user_bool = defaultdict(bool)

    for yr in range(2008, 2024):
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                v = list(line.values())[0]
                u_id = v[1]
                languagetemp = set(v[0]).intersection(programming_language_std_adjusted)
                if not user_bool[u_id]:
                    user_bool[u_id] = True
                    user_python_count[u_id] = [0,0]

                if 'python' in languagetemp:
                    user_python_count[u_id][0] += 1

                user_python_count[u_id][1] += 1

        fcc_file.close()

    python_user_bool = defaultdict(bool)
    user_python_share = defaultdict(float)
    for u_id, uc in user_python_count.items():
        user_python_share[u_id] = uc[0]/(uc[0] + uc[1])
        if uc[0]/(uc[0] + uc[1]) >= python_threshold:
            python_user_bool[u_id] = True

    save_obj(python_user_bool, 'python_user_bool_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(user_python_share, 'user_python_share_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    
    return




def collect_user_major_language_year(data_path_save, year_list):
    import random
    random.seed(1670)
    yr = year_list[0]
    user_language_collection_all = load_obj(f'user_language_count_by_single_year_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
    user_bool = defaultdict(bool)
    for u in user_language_collection_all.keys():
        user_bool[u] = True
    
    for yr in year_list[1:]:
        user_language_collection = load_obj(f'user_language_count_by_single_year_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, language_dict in user_language_collection.items():
            if not user_bool[u]:
                user_language_collection_all[u] = defaultdict(float)
                user_bool[u] = True

            for l,lc in language_dict.items():
                user_language_collection_all[u][l] += lc

    print(len(user_language_collection_all))

    
    user_major_language = {}
    user_major_language_all = {}
    for u, language_dict in user_language_collection_all.items():
        if len(language_dict) > 0:
            ltemp = sorted(language_dict.items(), key = lambda kv:(-kv[1], kv[0]))
            major_languages_temp = [lt[0] for lt in ltemp if lt[1] == ltemp[0][1]]
            user_major_language[u] = random.sample(major_languages_temp, 1)[0]
            user_major_language_all[u] = [l for l in major_languages_temp]

        else:
            user_major_language[u] = None
            user_major_language_all[u] = None

    save_obj(user_major_language, f'user_major_language_{year_list}', data_path_save + 'vote_regression_together/user_c_l_list/')
    save_obj(user_major_language_all, f'user_major_language_all_ties_{year_list}', data_path_save + 'vote_regression_together/user_c_l_list/')

#endregion


##? collect dataset
##! build dataframe for regression
#region - build jsonfile

def calculate_effective_number(data_list):

    if len(data_list) == 0:
        return 0

    sumdata = sum(data_list)
    prob_list = [-p/sumdata * np.log(p/sumdata) for p in data_list if p > 0]

    return np.exp(sum(prob_list))


def collect_answer_vote_coefficients(data_path, data_path_save, level, programming_language_std_adjusted, G,tag_bool, experience_period_length, language_len = [0,1,2,3,4,5,6,7,8,9,10], task_len = [0,1,2,3,4,5,6,7,8,9,10]):

    answer_upvote_count = load_obj('answer_upvote_count', data_path)
    user_python_share_global = load_obj('user_python_share_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    date_zero = date(2008,1,1)

    answer_rank_dict = load_obj(f'answer_rank_dict_alltime', data_path_save + 'vote_regression_together/user_c_l_list/')
    question_hour_dict = load_obj('question_hour_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    average_user_hour_global = load_obj('average_user_answer_hour_all_time', data_path_save + 'vote_regression_together/user_c_l_list/')
    median_user_hour_global = load_obj(f'median_user_answer_hour_all_time',  data_path_save + 'vote_regression_together/user_c_l_list/')
    answer_time_rank_dict = load_obj('answer_time_rank', data_path_save + 'vote_regression_together/user_c_l_list/')
    question_abs_minute_dict = load_obj('question_abs_minute_dict', data_path_save + 'vote_regression_together/user_c_l_list/')

    question_date = load_obj('question_date', data_path)
    

    for yr in range(2008 + experience_period_length + 1, 2024):
        print(yr)
        answer_parent_vote = load_obj(f'answer_parent_num_vote_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')

        user_answer_history_yr = load_obj(f'user_answer_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        
        user_rank = sorted(user_answer_history_yr.items(), key = lambda kv:(-kv[1], kv[0]))
        len_uc = len(user_rank)
        user_rank_dict = {uc[0]:uc[1]/len_uc for uc in user_rank}

        
        temp_answer_user = []
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                temp_answer_user.append(v[1])

        temp_answer_user = set(temp_answer_user)

        fcc_file.close()

        user_task_count = {u:defaultdict(float) for u in temp_answer_user}
        user_language_count = {u:defaultdict(float) for u in temp_answer_user}
        user_tl_count = {u:defaultdict(float) for u in temp_answer_user}
        user_answer_history = defaultdict(int)
        user_top1_answer_count = defaultdict(int)
        user_hour_lists = defaultdict(list)
        
        user_history_bool_dict = defaultdict()
        for epi in range(1, experience_period_length+1):
            user_task_count_temp = load_obj(f'user_task_count_by_single_year_level_{level}_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
            user_language_count_temp = load_obj(f'user_language_count_by_single_year_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
            user_tl_count_temp = load_obj(f'user_task_language_count_by_single_year_{yr - epi}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
            user_answer_history_temp = load_obj(f'user_answer_history_single_year_{yr - epi}',  data_path_save + 'vote_regression_together/user_c_l_list/')
            user_top_1_answer_count_temp = load_obj(f'user_top_1_answer_count_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
            user_hour_dict_temp =  load_obj(f'user_hour_dict_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')

            for u in temp_answer_user:
                if u in user_task_count_temp:
                    for task, task_count in user_task_count_temp[u].items():
                        user_task_count[u][task] += task_count
                
                if u in user_language_count_temp:
                    for language, language_count in user_language_count_temp[u].items():
                        user_language_count[u][language] += language_count

                if u in user_tl_count_temp:
                    for tl, tl_count in user_tl_count_temp[u].items():
                        user_tl_count[u][tl] += tl_count
                
                if u in user_answer_history_temp:
                    user_history_bool_dict[u] = True
                    user_answer_history[u] += user_answer_history_temp[u]

                user_top1_answer_count[u] += user_top_1_answer_count_temp[u]
                user_hour_lists[u] += user_hour_dict_temp[u]

        with jsonlines.open(f'{data_path_save}vote_regression_together/answer_vote_coefficients_json/answer_vote_coefficients_level_{level}_single_year_{yr}_period_{experience_period_length}.json', 'w') as w:
            with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'r') as fcc_file:
                for line in tqdm(fcc_file):
                    k = list(line.keys())[0]
                    v = list(line.values())[0]
                    ltemp = programming_language_std_adjusted.intersection(set(v[0]))
                    ctemp = set([G.nodes[t]['cluster_id'] for t in v[0] if tag_bool[t]])
                    l_set_len = len(ltemp)
                    c_set_len = len(ctemp)
                    u_id = v[1]
                    
                    if l_set_len in language_len and c_set_len in task_len:

                        prime_language = list(ltemp)[0]
                        for prime_task in ctemp:
                            task_efnum = calculate_effective_number(list(user_task_count[u_id].values()))
                            task_experience = user_task_count[u_id][prime_task]
                            language_efnum = calculate_effective_number(list(user_language_count[u_id].values()))
                            language_experience = user_language_count[u_id][prime_language]

                            user_answer_experience = user_answer_history[u_id]

                            ##! user是否出现在history里
                            if user_answer_experience > 0:
                                user_history_sign = 1
                            else:
                                user_history_sign = 0

                            answer_vote = answer_upvote_count[k] if k in answer_upvote_count else 0

                            other_task_experience = sum(user_task_count[u_id].values()) - task_experience
                            
                            other_language_experience = sum(user_language_count[u_id].values()) - language_experience

                            tle_temp = sum(user_language_count[u_id].values())
                            python_experience_share = user_language_count[u_id]['python'] / tle_temp if tle_temp > 0 else 0

                            alltime_python_experience_share = user_python_share_global[u_id]
                            answer_year = v[2][:4]
                            answer_month = v[2][5:7]
                            answer_day = v[2][8:10]
                            
                            answer_day_from_zero = (date(int(v[2][:4]), int(v[2][5:7]), int(v[2][8:10]))- date_zero).days

                            user_rank_this_year = user_rank_dict[u_id]
                            answer_rank = answer_rank_dict[k]
                            user_top1_answer_num = user_top1_answer_count[u_id]

                            question_hour = question_hour_dict[answer_parent_vote[k][3]]

                            answer_hour = int(v[2][11:13])
                            answer_minute = int(v[2][14:16])
                            answer_abs_minute = answer_hour * 60 + answer_minute
                            average_user_hour = np.mean(user_hour_lists[u_id]) if len(user_hour_lists[u_id]) > 0 else None
                            median_user_hour = np.median(user_hour_lists[u_id]) if len(user_hour_lists[u_id]) > 0 else None

                            average_user_hour_alltime = average_user_hour_global[u_id]
                            median_user_hour_alltime = median_user_hour_global[u_id]
                            answer_time_rank = answer_time_rank_dict[k]

                            question_abs_minute = question_abs_minute_dict[answer_parent_vote[k][3]]

                            task_language_count = user_tl_count[u_id][(prime_task, prime_language)]

                            question_year = question_date[answer_parent_vote[k][3]][:4]
                            question_month = question_date[answer_parent_vote[k][3]][5:7]
                            question_day = question_date[answer_parent_vote[k][3]][8:10]
                            question_day_from_zero = (date(int(question_year), int(question_month), int(question_day))- date_zero).days

                            ##! [0: task effect number, 1: language effect number, 2: parent_answer_num, 3: parent_answer_vote,
                            ##! 4: parent_vote, 5: parent_id, 6: task experience, 
                            ##! 7: language experience, 8: user answer history, 9: answer vote, 10: prime_task, 11: language,
                            ##! 12: other_language_experience, 13: other_task_experience, 14: user_id, 
                            ##! 15: user_python_experience_share 16: alltime_python_experience_share, 17: user_history_sign, 
                            ##! 18: answer_year, 19: answer_month, 20: answer_day, 21: answer_day_from_zero, 
                            ##! 22: user_rank_this_year, 23: answer_rank, 24: user_top1_answer_num,
                            ##! 25: question_hour, 26: answer_hour, 27: average_user_hour, 28: median_user_hour,
                            ##! 29: average_user_hour_alltime, 30: median_user_hour_alltime, 
                            ##! 31: answer_time_rank, 32: answer_minute 33: answer_abs_minute,
                            ##! 34: question_abs_minute, 35: task_language_experience,
                            ##! 36: question_year, 37: question_month, 38: question_day,
                            ##! 39: question_day_from_zero]
                            w.write({k:[task_efnum, language_efnum, answer_parent_vote[k][0], answer_parent_vote[k][1],
                                        answer_parent_vote[k][2], answer_parent_vote[k][3], task_experience, language_experience, user_answer_experience, answer_vote, prime_task, prime_language,
                                        other_language_experience, other_task_experience, u_id, 
                                        python_experience_share, alltime_python_experience_share, user_history_sign,
                                        answer_year, answer_month, answer_day, answer_day_from_zero,
                                        user_rank_this_year, answer_rank, user_top1_answer_num,
                                        question_hour, answer_hour, average_user_hour, median_user_hour,
                                        average_user_hour_alltime, median_user_hour_alltime,
                                        answer_time_rank, answer_minute, answer_abs_minute,
                                        question_abs_minute, task_language_count,
                                        question_year, question_month, question_day,
                                        question_day_from_zero]})

                    ##? update the user experience after collection

        w.close()

    return

#endregion



def vote_regression_build_dataframe(data_path_save, regression_year_list, experience_period_length, level, sample_user_label):

    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    
    data = []
    for yr in regression_year_list:
        with jsonlines.open(f'{data_path_save}vote_regression_together/answer_vote_coefficients_json/answer_vote_coefficients_level_{level}_single_year_{yr}_period_{experience_period_length}.json', 'r') as fcc_file:
            for line in tqdm(fcc_file):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                u_id = v[14]
                if sample_user_bool[u_id]:
                    data.append([k] + v)

        fcc_file.close()

    df = pd.DataFrame(data, columns=['answer_id', 'efnum_task', 'efnum_language', 'parent_ans_number', 'parent_answers_vote', 
                                    'parent_vote', 'parent_id', 'task_experience',
                                    'language_experience', 'user_answer_history', 'answer_vote', 'task', 'language',
                                    'other_language_experience', 'other_task_experience', 'user_id',
                                    'user_python_experience_share', 'alltime_user_experience_share', 'user_history_sign',
                                    'answer_year', 'answer_month', 'answer_day', 'answer_day_from_zero',
                                    'user_rank_this_year', 'answer_rank', 'user_top1_answer_num',
                                    'question_hour', 'answer_hour', 'average_user_answer_hour', 'median_user_answer_hour',
                                    'alltime_user_answer_hour_average', 'alltime_user_answer_hour_median',
                                    'answer_time_rank', 'answer_minute', 'answer_abs_minute',
                                    'question_abs_minute', 'task_language_experience',
                                    'question_year', 'question_month', 'question_day','question_day_from_zero'])
    

    return df


def collect_answer_task_minute_experience_build_dataframe(data_path_save, level, experience_period_length, programming_language_std_adjusted, G_tag_core, yr, sample_user_label):

    sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')

    #user_python_share_global = load_obj('user_python_share_dict', data_path_save + 'vote_regression_together/user_c_l_list/')
    #average_user_hour_global = load_obj('average_user_answer_hour_all_time', data_path_save + 'vote_regression_together/user_c_l_list/')
    #median_user_hour_global = load_obj(f'median_user_answer_hour_all_time',  data_path_save + 'vote_regression_together/user_c_l_list/')

    community_list = list(set([G_tag_core.nodes[n]['cluster_id'] for n in G_tag_core.nodes()]))
    
    print(yr)
    year_list_task = []
    year_list_language = []
    year_list_history = []
    abs_minute_list_task = []
    abs_minute_list_language = []
    abs_minute_list_history = []
    task_list = []
    language_list = []

    task_count_list = []
    language_count_list = []
    history_count_list = []

    task_count_list_log = []
    language_count_list_log = []
    history_count_list_log = []

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
    user_language_count = {u:defaultdict(float) for u in temp_answer_user}
    user_answer_history = defaultdict(int)

    for epi in range(1, experience_period_length+1):
        user_task_count_temp = load_obj(f'user_task_count_by_single_year_level_{level}_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        user_language_count_temp = load_obj(f'user_language_count_by_single_year_{yr - epi}', data_path_save + 'vote_regression_together/user_c_l_list/')
        user_answer_history_temp = load_obj(f'user_answer_history_single_year_{yr - epi}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u in temp_answer_user:
            if u in user_task_count_temp:
                for task, task_count in user_task_count_temp[u].items():
                    user_task_count[u][task] += task_count
            
            if u in user_language_count_temp:
                for language, language_count in user_language_count_temp[u].items():
                    user_language_count[u][language] += language_count
            
            if u in user_answer_history_temp:
                user_answer_history[u] += user_answer_history_temp[u]

    ##! 计算experience
    ##! 计算 user 的 task 和 language count
    print("calculate weighted experience")
    task_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    language_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    answer_history_by_abs_minute = {abs_minute:0 for abs_minute in range(1440)}

    log_task_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    log_language_experience_by_abs_minute = {abs_minute:defaultdict(float) for abs_minute in range(1440)}
    log_answer_history_by_abs_minute = {abs_minute:0 for abs_minute in range(1440)}

    for abs_minute in tqdm(range(1440)):
        for u in set(temp_answer_user):
            u_absminute_average = user_average_minute_dict[u]
            u_weight = min(1440 - np.abs(u_absminute_average - abs_minute), np.abs(u_absminute_average - abs_minute))
            u_weight = 720 - u_weight
            for task, task_count in user_task_count[u].items():
                task_experience_by_abs_minute[abs_minute][task] += u_weight * task_count
                log_task_experience_by_abs_minute[abs_minute][task] += np.log(u_weight * task_count + 1)
        
            for language, language_count in user_language_count[u].items():
                language_experience_by_abs_minute[abs_minute][language] += u_weight * language_count
                log_language_experience_by_abs_minute[abs_minute][language] += np.log(u_weight * language_count + 1)

            answer_history_by_abs_minute[abs_minute] += u_weight * user_answer_history[u]
            log_answer_history_by_abs_minute[abs_minute] += np.log(u_weight * user_answer_history[u] + 1)


    print("collect data for dataframes")
    for abs_minute in range(1440):
        for task in community_list:
            year_list_task.append(yr)
            task_list.append(task)
            task_count_list.append(task_experience_by_abs_minute[abs_minute][task])
            task_count_list_log.append(log_task_experience_by_abs_minute[abs_minute][task])
            abs_minute_list_task.append(abs_minute)

        for language in programming_language_std_adjusted:
            year_list_language.append(yr)
            language_list.append(language)
            language_count_list.append(language_experience_by_abs_minute[abs_minute][language])
            language_count_list_log.append(log_language_experience_by_abs_minute[abs_minute][language])
            abs_minute_list_language.append(abs_minute)

        year_list_history.append(yr)
        history_count_list.append(answer_history_by_abs_minute[abs_minute])
        history_count_list_log.append(log_answer_history_by_abs_minute[abs_minute])
        abs_minute_list_history.append(abs_minute)

    df_absminute_task = pd.DataFrame.from_dict({'year':year_list_task, 'task': task_list, 'weighted_task_experience':task_count_list, 'weighted_task_experience_log': task_count_list_log,'minute_of_the_day': abs_minute_list_task})
    df_absminute_language = pd.DataFrame.from_dict({'year':year_list_language, 'language': language_list, 'weighted_language_experience':language_count_list, 'weighted_language_experience_log': language_count_list_log, 'minute_of_the_day': abs_minute_list_language})
    df_absminute_history = pd.DataFrame.from_dict({'year':year_list_history, 'weighted_answer_history_experience':history_count_list, 'weighted_answer_history_experience_log':history_count_list_log, 'minute_of_the_day': abs_minute_list_history})

    df_absminute_history.to_csv(data_path_save + f'vote_regression_together/df_year/{sample_user_label}_answer_history_abs_minute_dataframe_{yr}.csv')
    df_absminute_task.to_csv(data_path_save + f'vote_regression_together/df_year/{sample_user_label}_task_abs_minute_dataframe_{yr}.csv')
    df_absminute_language.to_csv(data_path_save + f'vote_regression_together/df_year/{sample_user_label}_language_abs_minute_dataframe_{yr}.csv')

    return


def merge_dataframe(data_path_save, experience_period_length, sample_user_label):
    df_list_history = []
    df_list_task = []
    df_list_language = []
    for yr in range(2008 + experience_period_length + 1, 2024):
        df_history = pd.read_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_answer_history_abs_minute_dataframe_{yr}.csv')
        df_list_history.append(df_history)
        df_task = pd.read_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_task_abs_minute_dataframe_{yr}.csv')
        df_list_task.append(df_task)
        df_language = pd.read_csv(data_path_save + f'vote_regression_together/df_year_parallel/{sample_user_label}_language_abs_minute_dataframe_{yr}.csv')
        df_list_language.append(df_language)

    df_history_all = pd.concat(df_list_history)
    df_task_all = pd.concat(df_list_task)
    df_language_all = pd.concat(df_list_language)
    
    df_history_all.to_csv(data_path_save + f'vote_regression_together/{sample_user_label}_history_abs_minute_dataframe.csv')
    df_task_all.to_csv(data_path_save + f'vote_regression_together/{sample_user_label}_task_abs_minute_dataframe.csv')
    df_language_all.to_csv(data_path_save + f'vote_regression_together/{sample_user_label}_language_abs_minute_dataframe.csv')

    return
