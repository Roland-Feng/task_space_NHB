import graph_tool.all as gt
import numpy as np
import random
import pandas as pd
import sbm_bipartite
import pickle
import time

np.random.seed(43)
seed=43
gt.seed_rng(seed) ## seed for graph-tool's random number generator --> same results

###############################################################################################
###############################################################################################

def save_obj(obj, name, data_path_save = 'obj/'):
    with open(data_path_save + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def load_obj(name, data_path_load = 'obj/'):
    with open(data_path_load + name + '.pkl', 'rb') as f:
        return pickle.load(f)

print('loading graph...')

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!图文件名

label = 'question_tag_bipartite'
model_name = data_path_save + 'question_tag_bipartite_graph'

######################################################################################


model=sbm_bipartite.bipartite_sbm()
model.load_model(model_name)

def block_dict_to_dataframe(block_dict, tag_count):

    tag_count_dict = {t[0]:t[1] for t in tag_count}

    tree_length = len(list(block_dict.values())[0]) 
    nodelist_temp = [n for n in list(block_dict.keys())]
    nodelist_count = [tag_count_dict[n] for n in nodelist_temp]

    b, nodelist = zip(*sorted(zip(nodelist_count,nodelist_temp), reverse=True))
    
    block_level_dict = {"TAG":nodelist}
    levels_len = []

    for i in range(tree_length):
        block_level_dict['level_'+str(i)] = []
        temp = []
        for n in nodelist:
            l = block_dict[n]
            if l[i] not in temp:
                temp.append(l[i])

        temp_dict = {b:bi for bi,b in enumerate(temp)}
        for n in nodelist:
            block_level_dict['level_'+str(i)].append(temp_dict[block_dict[n][i]])

        levels_len.append(len(temp))

    return pd.DataFrame.from_dict(block_level_dict), levels_len


levels = model.state.get_levels()
levels_len = len(levels)

tag_map_from_gt_to_nx = {}

for i in model.g.vertices():
    if model.g.vp.kind[i] == 1:
        tag_map_from_gt_to_nx[i] = model.g.vp.name[i]
        
        
        
block_dict = {}
for v,nxv in tag_map_from_gt_to_nx.items():
    block_dict[nxv] = [levels[0].get_blocks()[v]]
    for l in range(1,levels_len):
        block_dict[nxv].append(levels[l].get_blocks()[block_dict[nxv][-1]])
        
        
tag_count = load_obj('tag_count_all', data_path_save)

df_block, level_len = block_dict_to_dataframe(block_dict, tag_count)

print(level_len)

save_obj(df_block, f'{label}_df_nested_sbm_block_unweighted', data_path_save + 'communities/')
save_obj(level_len, f'{label}_levels_nested_sbm_block_unweighted', data_path_save  + 'communities/')