import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
import random
from CoLoc_class import CoLoc #import the CoLoc class
import scipy

##! K:[0 tags, 1 user, 2 date, 3 answer]
def sample_from_users(year_bool, data_path_save, sample_ratio, sample_threshold):

    user_answer_number_dict = defaultdict(int)
    year_list = [yr for yr, yrb in year_bool.items() if yrb]
    print(year_list)
    for yr in tqdm(year_list):
        user_answer_history = load_obj(f'user_answer_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, uah in user_answer_history.items():
            user_answer_number_dict[u] += uah

    threshold_user_bool = defaultdict(bool)
    for u, uah in user_answer_number_dict.items():
        if uah >= sample_threshold:
            threshold_user_bool[u] = True

    random.seed(43)
    sample_user_bool = defaultdict(bool)
    user_list = list(set([u for u, ub in threshold_user_bool.items() if ub]))
    user_sample = random.sample(user_list, int(len(user_list) * sample_ratio))
    for u in user_sample:
        sample_user_bool[u] = True

    return sample_user_bool, threshold_user_bool

def get_all_user_bool(data_path_save, qa):

    all_answer_user_bool_dict = defaultdict(bool)
    for yr in tqdm(range(2008,2024)):
        user_answer_history = load_obj(f'user_{qa}_history_single_year_{yr}',  data_path_save + 'vote_regression_together/user_c_l_list/')
        for u in user_answer_history.keys():
            all_answer_user_bool_dict[u] = True

    return all_answer_user_bool_dict


def get_user_community_set_from_sample(year_bool, user_bool, data_path_save,community_list_std, level):

    C_dict = {c:i for i,c in enumerate(community_list_std)}

    ##! 收集user tag
    tasks_user_all_set = defaultdict(set)
    yr_list = [yr for yr, yb in year_bool.items() if yb]
    for yr in tqdm(yr_list):
        task_user_set = load_obj(f'task_user_set_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u,uts in task_user_set.items():
            if user_bool[u]:
                tasks_user_all_set[u] = tasks_user_all_set[u] | uts


    ##! 收集user community
    user_community_set = {}
    user_community_vector_binary = {}
    for u,uc_set in tqdm(tasks_user_all_set.items()):
        if len(uc_set) > 0:
            user_community_set[u] = [i for i in uc_set]
            user_community_vector_binary[u] = [0 for c in community_list_std]
            for c in uc_set:
                user_community_vector_binary[u][C_dict[c]] = 1

    return user_community_set, user_community_vector_binary


def get_bayesian_pmi_edgelist_from_occurrence(co_occurrrence, p_value = 0.05):
    co_df = pd.DataFrame({"c1":[a[0] for a in co_occurrrence] + [a[1] for a in co_occurrrence], "c2":[a[1] for a in co_occurrrence] + [a[0] for a in co_occurrrence], "cc":[a[2] for a in co_occurrrence] + [a[2] for a in co_occurrrence]})
    
    dft = co_df.pivot(index = 'c1', columns = 'c2', values = 'cc')
    dft = dft.fillna(0)
    
    #Q = CoLoc(df_q, prior = 'uniform', nr_prior_obs = np.size(df_q))
    Q = CoLoc(dft)

    df_Q = Q.make_sigPMIpci(p_value)

    df_index = df_Q.index
    df_columns = df_Q.columns

    res = scipy.sparse.coo_matrix(df_Q.fillna(0).values)

    df_res = pd.DataFrame({'c1':df_columns[res.col], 'c2':df_index[res.row], 'cc':res.data})

    df_edgelist = df_res[df_res['cc'] > 0]

    edgelist_sig = [(c1,c2,cc) for c1,c2,cc in zip(df_edgelist['c1'], df_edgelist['c2'], df_edgelist['cc']) if c1 != c2]

    #df_variance = Q.make_stdPMIpci()

    return edgelist_sig, Q



##! community_cooccurrence matrix
def build_community_cooccurrence(user_community_set, community_list_std, p_value):

    C_dict = {c:i for i,c in enumerate(community_list_std)}

    ##! 收集community cooccurrence
    community_co_occurrence = defaultdict(int)
    for u, c_set in tqdm(user_community_set.items()):
        if len(c_set) > 1:
            cc_temp = [(c1, c2) for c1 in c_set for c2 in c_set if c1 < c2]
            for cc in cc_temp:
                community_co_occurrence[cc] += 1

    cc_cooccurrence = [(cc[0],cc[1],oc) for cc,oc in community_co_occurrence.items()]

    print(f"p value: {p_value}")
    cc_pmi, Q =get_bayesian_pmi_edgelist_from_occurrence(cc_cooccurrence, p_value)
    
    cc_pmi_matrix = np.zeros((len(community_list_std), len(community_list_std)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    for i in range(len(community_list_std)):
        if cc_pmi_matrix[i,:].sum() != 0:
            cc_pmi_matrix[i,:] = cc_pmi_matrix[i,:] / cc_pmi_matrix[i,:].sum()

    return cc_pmi, cc_pmi_matrix, Q


##! make prediction
def predict_user_community(cc_pmi_matrix, user_community_vector_binary):
    user_prediction = {}
    for u,vb in tqdm(user_community_vector_binary.items()):
        user_prediction[u] = cc_pmi_matrix.dot(np.array(vb)).T

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
