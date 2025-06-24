import pandas as pd
import networkx as nx
import numpy as np
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from random import sample
from CoLoc_class import CoLoc #import the CoLoc class
import scipy
from pickle_file import load_obj, save_obj

##Read and Load files

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


def get_community_at_level(df_levels, level_index):
    levels_community = {str(ci):[] for ci in df_levels[level_index]}
    for t,ci in zip(df_levels['TAG'], df_levels[level_index]):
        levels_community[str(ci)].append(t)

    return levels_community


def get_user_community_df(year_list, level):
    
    community_list_std =  load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    threshold_user_bool = load_obj('all_threshold_user_bool', data_path_save + f'vote_regression_together/')

    C_dict = {c:i for i,c in enumerate(community_list_std)}

    user_community_vector = {}
    user_bool = defaultdict(bool)
    for yr in tqdm(year_list):
        user_task_count = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + f'vote_regression_together/user_c_l_list/')
        for u,tc_dict in user_task_count.items():
            if threshold_user_bool[u]:
                if not user_bool[u]:
                    user_community_vector[u] = [0 for c in community_list_std]
                    user_bool[u] = True

                for c, ctc in tc_dict.items():
                    user_community_vector[u][C_dict[c]] += ctc

    community_list_std_name = ['community_'+i for i in community_list_std]
    user_community_df  = pd.DataFrame.from_dict(user_community_vector, orient = 'index', columns=community_list_std_name)

    return user_community_df




def get_task_task_density(level):
    cc_pmi = load_obj(f'cc_pmi_all_threshold_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    community_list_std =  load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std_name = ['community_' + i for i in community_list_std]
    C_dict = {c:i for i,c in enumerate(community_list_std)}
    cc_pmi_matrix = np.zeros((len(community_list_std), len(community_list_std)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    df = pd.DataFrame(data=cc_pmi_matrix, columns=community_list_std_name,index=community_list_std_name)

    return df, min([ccp[2] for ccp in cc_pmi])



def get_task_task_density_no_empty(level):

    community_list_std =  load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    
    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std_no_empty = [c for c in community_list_std if len(community_unweighted_level[c]) > 0]
    C_dict = {c:i for i,c in enumerate(community_list_std_no_empty)}

    cc_pmi = load_obj(f'cc_pmi_all_threshold_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    
    community_list_std_name = ['community_' + i for i in community_list_std_no_empty]
    C_dict = {c:i for i,c in enumerate(community_list_std_no_empty)}
    cc_pmi_matrix = np.zeros((len(community_list_std_no_empty), len(community_list_std_no_empty)))
    for ccp in cc_pmi:
        cc_pmi_matrix[C_dict[ccp[0]], C_dict[ccp[1]]] = ccp[2]
        cc_pmi_matrix[C_dict[ccp[1]], C_dict[ccp[0]]] = ccp[2]

    df = pd.DataFrame(data=cc_pmi_matrix, columns=community_list_std_name,index=community_list_std_name)

    return df, min([ccp[2] for ccp in cc_pmi])

def get_task_size(year_list, level):
    community_list_std =  load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    C_dict = {c:i for i,c in enumerate(community_list_std)}
    task_size = defaultdict(int)
    for yr in year_list:
        task_language_matrix = load_obj(f'answer_task_language_matrix_{[yr]}_level_{level}', data_path_save + 'task_language_nestedness/tl_matrix/')
        for c in community_list_std:
            task_size[c] += np.sum(task_language_matrix[C_dict[c], :])

    return task_size


def points_to_data_units(fig, points, axis):
    # Get the figure DPI
    dpi = fig.dpi
    # Get the axis size in pixels
    ax_size_in_pixels = axis.bbox.size
    # Calculate points to pixels
    points_in_pixels = points * dpi / 72
    # Convert pixels to data units
    data_unit_per_pixel_x = (axis.get_xlim()[1] - axis.get_xlim()[0]) / ax_size_in_pixels[0]
    data_unit_per_pixel_y = (axis.get_ylim()[1] - axis.get_ylim()[0]) / ax_size_in_pixels[1]
    # Return data units
    
    return points_in_pixels * data_unit_per_pixel_x/2, points_in_pixels * data_unit_per_pixel_y/2
