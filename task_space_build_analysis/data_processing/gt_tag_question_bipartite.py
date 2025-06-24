import graph_tool.all as gt
import numpy as np
import random
import pandas as pd
import sbm_bipartite
import pickle
import time

import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

data_path = '/home/xiangnan/task_space_code/task_space_data/'
#data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_bool = load_obj('tag_bool_threshold_adjusted', data_path_save)


##! build community colocation matrix
year_bool = {str(yr):False for yr in range(2008, 2024)}
for yr in range(2008,2024):
    year_bool[str(yr)] = True

print('================================= tag pair =================================')
csv_file_path = build_tag_question_bipartite(year_bool, data_path_save, 'question', tag_bool)


dest = csv_file_path

np.random.seed(43)
seed=43
gt.seed_rng(seed) ## seed for graph-tool's random number generator --> same results

print('loading graph...')
model=sbm_bipartite.bipartite_sbm()
model.load_graph(dest)

print('finding communities with sbm ...')
model.fit()
model.state

model.save_model(data_path_save + 'question_tag_bipartite_graph')


###############################################################################################
###############################################################################################
