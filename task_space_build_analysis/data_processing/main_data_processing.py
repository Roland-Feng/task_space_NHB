###########################################################################################
##! task synonyms and save tag set
###########################################################################################

from pickle_file import save_obj, load_obj
data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'
import pandas as pd

tag_synonyms_dict = {}
df = pd.read_csv(data_path + 'tag_synonyms.csv')
df_dict = pd.Series(df.TargetTagName.values,index=df.SourceTagName).to_dict()
tag_synonyms_dict = {k.lower().replace('-','_'):v.lower().replace('-','_') for k,v in df_dict.items() if not pd.isna(v) and v[:10] != 'do-not-use'}
tag_synonyms_dict['nulls'] = 'null'
tag_synonyms_dict['nullvalue'] = 'null'
tag_synonyms_dict['nil'] = 'null'

problem_tags_temp = [str(k).lower().replace('-','_') for k,v in df_dict.items() if pd.isna(v) or v[:10] == 'do-not-use']
problem_tags = ['state_managment', 'stars']

save_obj(tag_synonyms_dict, 'tag_synonyms_dict', data_path)

###########################################################################################

import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tags_with_sort_1 = pd.read_csv(data_path + "tags_larger_than_11.csv")
tags_with_sort_2 = pd.read_csv(data_path + "tags_less_than_11.csv")
tags_all = pd.concat([tags_with_sort_1, tags_with_sort_2])
taglist_all_temp = [str(t).lower().replace('-','_') for t in tags_all['TagName'].values.tolist()]

tag_synonyms_dict = load_obj('tag_synonyms_dict', data_path_save)

#! 保存 tag 的rename_dict

tag_rename_dict = {k:v for k,v in tag_synonyms_dict.items()}

save_obj(tag_rename_dict,'tag_rename_dict',data_path)

#! tag_set
tag_need_rename = set(taglist_all_temp).intersection(set(tag_rename_dict.keys()))
tag_need_no_rename = set(taglist_all_temp) - tag_need_rename

tagset_all = tag_need_no_rename | set([tag_rename_dict[t] for t in tag_need_rename]) - set(problem_tags)

save_obj(tagset_all, 'tagset_all', data_path)


###########################################################################################
##! reading tags
###########################################################################################
# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_answer_list = {}
q_tags_bool = {}
with jsonlines.open(f'{data_path}all_so_posts.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        q_tags_bool[k] = False
        ##! 限制条件
        if v[0] == '1' and len(k) > 0 and len(v[5]) > 0:
            question_answer_list[k] = []
            q_tags_bool[k] = True

fcc_file.close()


with jsonlines.open(f'{data_path}all_so_posts.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]

        if v[0] == '2' and len(k) > 0 and len(v[5]) > 0 and len(v[10]) > 0 and q_tags_bool[v[10]]:
            question_answer_list[v[10]].append(k)

fcc_file.close()


tag_rename_dict = load_obj('tag_rename_dict', data_path_save)
tagset_all = load_obj('tagset_all', data_path_save)

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'w') as w:
    with jsonlines.open(f'{data_path}all_so_posts.json', 'r') as fcc_file:
        for line in tqdm(fcc_file):
            k = list(line.keys())[0]
            v = list(line.values())[0]
            if q_tags_bool[k]:
                tag_list_temp1 = v[7].lstrip('<').rstrip('>').split('><')
                tag_list_temp = [t.lower().replace('-','_') for t in tag_list_temp1]
                tag_set = set([t for t in tag_list_temp if tag_rename_dict.get(t) is None] + [tag_rename_dict[t] for t in tag_list_temp if tag_rename_dict.get(t) is not None]).intersection(tagset_all)
                if len(tag_set) > 0:
                    w.write({k:[list(tag_set), v[5], v[2], question_answer_list[k]]})

fcc_file.close()

w.close()


###########################################################################################
# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

##! 初始化
answer_date = {}
answer_user = {}

with jsonlines.open(f'{data_path}all_so_posts.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        if v[0] == '2' and len(k) > 0 and len(v[5]) > 0 and len(v[10]) > 0:
            answer_date[k] = v[2]
            answer_user[k] = v[5]

fcc_file.close()

##! 存储answer json文件
with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc:
    with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'w') as w:
        for line in tqdm(fcc):
            k = list(line.keys())[0]
            v = list(line.values())[0]
            for ka in v[3]:
                w.write({ka:[v[0], answer_user[ka], answer_date[ka], k]})
                
    w.close()
fcc.close()

###########################################################################################
# 
##! question
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

for yr in range(2008,2024):
    y = str(yr)
    with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc:
        with jsonlines.open(f'{data_path}all_question_so_tags_{yr}.json', 'w') as w:
            for line in tqdm(fcc):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if v[2][:4] == y:
                    w.write({k:v})
                    
        w.close()
    fcc.close()



##! answer
for yr in range(2008,2024):
    y = str(yr)
    with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'r') as fcc:
        with jsonlines.open(f'{data_path}all_answer_so_tags_{yr}.json', 'w') as w:
            for line in tqdm(fcc):
                k = list(line.keys())[0]
                v = list(line.values())[0]
                if v[2][:4] == y:
                    w.write({k:v})
                    
        w.close()
    fcc.close()


###########################################################################################
from pickle_file import load_obj, save_obj
import json
import networkx as nx
from tqdm import tqdm
import jsonlines

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_list = []

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        tag_list += v[0]

fcc_file.close()

from collections import Counter
tag_count_temp = dict(Counter(tag_list))
tag_count = sorted(tag_count_temp.items(), key = lambda kv:(-kv[1], kv[0]))

save_obj(tag_count,'tag_count_all', data_path)

from pickle_file import load_obj, save_obj
import json
import networkx as nx
from tqdm import tqdm
import jsonlines

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_count = load_obj('tag_count_all', data_path)
tag_list = [t[0] for t in tag_count]

save_obj(tag_list,'tag_list_all', data_path)


###########################################################################################
# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_user = {str(yr):[] for yr in range(2008,2024)}
answer_user = {str(yr):[] for yr in range(2008,2024)}

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        question_user[v[2][:4]].append(v[1])
        
fcc_file.close()

with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        answer_user[v[2][:4]].append(v[1])

fcc_file.close()

question_user_temp = {yr:set(ul) for yr, ul in question_user.items()}
answer_user_temp = {yr:set(ul) for yr, ul in answer_user.items()}

save_obj(question_user_temp, 'question_user_by_year', data_path)
save_obj(answer_user_temp, 'answer_user_by_year', data_path)


# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_list = {str(yr):[] for yr in range(2008,2024)}
answer_list = {str(yr):[] for yr in range(2008,2024)}

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        question_list[v[2][:4]].append(k)
        
fcc_file.close()

with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        answer_list[v[2][:4]].append(k)

fcc_file.close()

save_obj(question_list, 'question_list_by_year', data_path)
save_obj(answer_list, 'answer_list_by_year', data_path)

###########################################################################################
# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_list = load_obj('question_list_by_year', data_path)
answer_list = load_obj('answer_list_by_year', data_path)

from collections import defaultdict

question_bool = defaultdict(bool)

for ql in tqdm(list(question_list.values())):
    for q in ql:
        question_bool[q] = True


answer_bool = defaultdict(bool)

for ql in tqdm(list(answer_list.values())):
    for q in ql:
        answer_bool[q] = True

save_obj(question_bool, 'question_bool_dict', data_path)
save_obj(answer_bool, 'answer_bool_dict', data_path)

###########################################################################################
# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_answer_list = {}

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        question_answer_list[k] = len(v[3])

fcc_file.close()

save_obj(question_answer_list, 'question_answer_list', data_path)

# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

answer_question_dict = {}

with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        answer_question_dict[k] = v[3]

fcc_file.close()

save_obj(answer_question_dict, 'answer_question_dict', data_path)

###########################################################################################

# python的qa的user 在 所有的数据上的 question和answer的tag
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

question_date = {}

with jsonlines.open(f'{data_path}all_question_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        question_date[k] = v[2]

fcc_file.close()

save_obj(question_date, 'question_date', data_path)

del question_date

answer_date = {}

with jsonlines.open(f'{data_path}all_answer_so_tags.json', 'r') as fcc_file:
    for line in tqdm(fcc_file):
        k = list(line.keys())[0]
        v = list(line.values())[0]
        answer_date[k] = v[2]

fcc_file.close()

save_obj(answer_date, 'answer_date', data_path)

###########################################################################################
import jsonlines
import random
from pickle_file import load_obj, save_obj
from tqdm import tqdm
import json

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

so_year_list = [yr for yr in range(2008,2024)]
user_answer_time = {}

for yr in so_year_list:
    print(yr)
    with jsonlines.open(f'{data_path}all_answer_so_tags_{str(yr)}.json', 'r') as fcc_file:
        for line in tqdm(fcc_file):
            v = list(line.values())[0]
            if user_answer_time.get(v[1]) is None:
                user_answer_time[v[1]] = [0 for h in range(24)]

            user_answer_time[v[1]][int(v[2][11:13])] += 1

    fcc_file.close()

save_obj(user_answer_time,'user_answer_time_hour', data_path)

###########################################################################################
##! programming languages
###########################################################################################
import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
import numpy as np
from collections import defaultdict

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_count = load_obj('tag_count_all', data_path)
tag_count_threshold = 1000

dtype = {"programming_language":str, "tag_count":str, "keep":str}
pl_filtered = pd.read_csv(data_path_save + 'programming languages filtered - Sheet1.csv', dtype = dtype).dropna()


dtype = {"TAG":str, "tag_count":str, "keep":str}
nonpl_filtered = pd.read_csv(data_path_save + 'non programming language - filtered - Sheet1.csv', dtype = dtype).dropna()


dtype = {"LF_TAG":str, "tag_count":str, "keep":str}
LFtag_filtered = pd.read_csv(data_path_save + 'less frequent tags filtered - Sheet1.csv', dtype = dtype).dropna()

programming_language_std_adjusted = []
for pl, k in zip(pl_filtered['programming_language'], pl_filtered['keep']):
    if k == '0':
        programming_language_std_adjusted.append(pl)


for pl, k in zip(nonpl_filtered['TAG'], nonpl_filtered['keep']):
    if k == '0':
        programming_language_std_adjusted.append(pl)
        

tag_bool_adjusted = {t[0]:False for t in tag_count}
for tc in tag_count:
    if tc[1] > tag_count_threshold and tc[0] not in programming_language_std_adjusted:
        tag_bool_adjusted[tc[0]] = True


for lftag, k in zip(LFtag_filtered['LF_TAG'], LFtag_filtered['keep']):
    if k == '1':
        tag_bool_adjusted[lftag] = True

save_obj(tag_bool_adjusted, 'tag_bool_threshold_adjusted', data_path_save)
save_obj(programming_language_std_adjusted, 'programming_language_std_adjusted', data_path_save)


###########################################################################################
##! SBM tasks
###########################################################################################
import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_bool = load_obj('tag_bool_threshold_adjusted', data_path_save)


##! build community colocation matrix
year_bool = {str(yr):False for yr in range(2008, 2024)}
for yr in range(2008,2024):
    year_bool[str(yr)] = True

print('================================= tag pair =================================')
csv_file_path = build_tag_question_bipartite(year_bool, data_path, data_path_save, 'question', tag_bool)

###########################################################################################
import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_bool = load_obj('tag_bool_threshold_adjusted', data_path_save)

##! build community colocation matrix
year_bool = {str(yr):False for yr in range(2008, 2024)}
for yr in range(2008,2024):
    year_bool[str(yr)] = True

print('================================= tag pair =================================')
tag_pair_index_count_posterior, _, index_tag_dict_posterior = get_tag_pair(year_bool, data_path, 'question', tag_bool)

print('================================= build network and community =================================')
G_tag_posterior, Q_posterior = build_network_from_tag_pair_posterior(tag_pair_index_count_posterior, index_tag_dict_posterior)
tag_bool_G = update_tag_bool(tag_bool,G_tag_posterior)


save_obj(G_tag_posterior, "G_tag_posterior", data_path_save + 'networks/')
save_obj(tag_bool_G, "tag_bool_adjusted_network", data_path_save + 'networks/')


###########################################################################################
import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
import numpy as np
from collections import defaultdict

import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

label = 'question_tag_bipartite'
df_unweight = load_obj(f'{label}_df_nested_sbm_block_unweighted', data_path_save + 'communities/')
levels_unweight = load_obj(f'{label}_levels_nested_sbm_block_unweighted', data_path_save + 'communities/')
community_unweighted_level0 = get_community_at_level(df_unweight, "level_0")
community_unweighted_level1 = get_community_at_level(df_unweight, "level_1")
community_unweighted_level2 = get_community_at_level(df_unweight, "level_2")
community_unweighted_level3 = get_community_at_level(df_unweight, "level_3")

G_tag_posterior = load_obj("G_tag_posterior", data_path_save + 'networks/')
G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level0)
save_obj(G_tag_posterior, f"G_tag_posterior_level0", data_path_save + 'networks/')
community_list_std = [i for i,c in community_unweighted_level0.items()]
save_obj(community_list_std, 'community_list_std_level_0', data_path_save + 'networks/')
save_obj(community_unweighted_level0, 'community_unweighted_level0', data_path_save + 'networks/')

G_tag_posterior = load_obj("G_tag_posterior", data_path_save + 'networks/')
G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level1)
save_obj(G_tag_posterior, f"G_tag_posterior_level1", data_path_save + 'networks/')
community_list_std = [i for i,c in community_unweighted_level1.items()]
save_obj(community_list_std, 'community_list_std_level_1', data_path_save + 'networks/')
save_obj(community_unweighted_level1, 'community_unweighted_level1', data_path_save + 'networks/')

G_tag_posterior = load_obj("G_tag_posterior", data_path_save + 'networks/')
G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level2)
save_obj(G_tag_posterior, f"G_tag_posterior_level2", data_path_save + 'networks/')
community_list_std = [i for i,c in community_unweighted_level2.items()]
save_obj(community_list_std, 'community_list_std_level_2', data_path_save + 'networks/')
save_obj(community_unweighted_level2, 'community_unweighted_level2', data_path_save + 'networks/')

G_tag_posterior = load_obj("G_tag_posterior", data_path_save + 'networks/')
G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level3)
save_obj(G_tag_posterior, f"G_tag_posterior_level3", data_path_save + 'networks/')
community_list_std = [i for i,c in community_unweighted_level3.items()]
save_obj(community_list_std, 'community_list_std_level_3', data_path_save + 'networks/')
save_obj(community_unweighted_level3, 'community_unweighted_level3', data_path_save + 'networks/')
###########################################################################################
import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

tag_bool = load_obj('tag_bool_threshold_adjusted', data_path_save)

##! build community colocation matrix
year_bool = {str(yr):False for yr in range(2008, 2024)}
for yr in range(2008,2024):
    year_bool[str(yr)] = True

print('================================= tag pair =================================')
tag_pair_index_count_posterior, _, index_tag_dict_posterior = get_tag_pair(year_bool, data_path, 'question', tag_bool)

print('================================= build network and community =================================')
G_tag_posterior = build_network_from_tag_cooccurrence(tag_pair_index_count_posterior, index_tag_dict_posterior)
tag_bool_G = update_tag_bool(tag_bool,G_tag_posterior)

save_obj(G_tag_posterior, "G_tag_cooccurrence", data_path_save + 'networks/probability/')
save_obj(tag_bool_G, "tag_bool_adjusted_network_cooccurrence", data_path_save + 'networks/probability/')


label = 'question_tag_bipartite'
df_unweight = load_obj(f'{label}_df_nested_sbm_block_unweighted', data_path_save + 'communities/')
levels_unweight = load_obj(f'{label}_levels_nested_sbm_block_unweighted', data_path_save + 'communities/')
for i in range(4):
    community_unweighted_level = get_community_at_level(df_unweight, f"level_{i}")
    G_tag_posterior = load_obj("G_tag_cooccurrence", data_path_save + 'networks/probability/')
    G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level)
    save_obj(G_tag_posterior, f"G_tag_cooccurrence_level{i}", data_path_save + 'networks/probability/')
###########################################################################################
import importlib
import tag_network_posterior #import the module here, so that it can be reloaded.
importlib.reload(tag_network_posterior)
from tag_network_posterior import *

#data_path = '/home/xiangnan/SO_code/'
data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

for level in range(1,4):
    print(level)
    community_unweighted_level = load_obj(f'community_unweighted_level{level}', data_path_save + 'networks/')
    G_tag_posterior = load_obj(f"G_tag_cooccurrence_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f'community_list_std_level_{level}', data_path_save + 'networks/')

    ##! calculate the RCA
    rca_matrix, tag_list_std, rca_values = get_tag_community_rca(G_tag_posterior, community_list_std, community_unweighted_level)
    community_tags, core_rca_sorted, empty_community = get_tags_rca_in_community(rca_values, community_unweighted_level, -1)

    ##! core bool
    cut_length = int(0.2 * len(core_rca_sorted))
    core_bool_temp = defaultdict(bool)
    core_bool = defaultdict(bool)
    for tc in core_rca_sorted[cut_length:]:
        core_bool_temp[tc[0][0]] = True
    
    ##! community core with cut
    community_cores_with_cut = {c:[] for c in community_list_std}
    for c, ts in community_tags.items():
        community_cores_with_cut[c] = [t for t in ts if core_bool_temp[t]]
        if len(community_cores_with_cut[c]) < 3:
            community_cores_with_cut[c] = []

        for t in community_cores_with_cut[c]:
            core_bool[t] = True

    ##! save dataframe
    df_dict = {'community':community_list_std, 'tags':[community_unweighted_level[c] for c in community_list_std], 'core_tags':[community_cores_with_cut[c] for c in community_list_std]}
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(data_path_save + f'networks/probability/core_rca_with_cut_level_{level}.csv')

    ##! save files
    save_obj(community_list_std, f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    save_obj(community_cores_with_cut, f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    G_tag_posterior = color_network_clusters(G_tag_posterior, community_unweighted_level)
    G_tag_core_with_cut = G_tag_posterior.subgraph([t for ts in community_cores_with_cut.values() for t in ts])
    save_obj(G_tag_core_with_cut, f"G_tag_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    save_obj(core_bool, f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')

    core_tags = [t for ts in community_cores_with_cut.values() for t in ts]
    core_tags_df = pd.DataFrame.from_dict({'core_tags': core_tags})
    core_tags_df.to_csv(data_path_save + f'networks/probability/core_tags_list_level_{level}.csv')


    for c, ts in community_cores_with_cut.items():
        if len(ts) == 0:
            print(c,ts)
###########################################################################################
##! get task names
###########################################################################################

from pickle_file import load_obj, save_obj

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {tc[0]:tc[1] for tc in tag_count}


level = 1
community_cores_with_cut1 = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')

community_cores_with_cut1_sorted = {c:[] for c in community_cores_with_cut1.keys()}
for c, ts in community_cores_with_cut1.items():
    tsc = [tag_count_dict[t] for t in ts]
    if len(tsc) > 0:
        tsc, ts = zip(*sorted(zip(tsc,ts), reverse=True))
        community_cores_with_cut1_sorted[c] = [t for t in ts]
    

# from openai import OpenAI
# from tqdm import tqdm

# task_unweighted_level1_general = {}
# ## input your key
# client = OpenAI(api_key='sk-proj-o7Whorgu')

# for ci in tqdm(community_cores_with_cut1_sorted.keys()):
    
#     response = client.chat.completions.create(
#       model="gpt-4-0125-preview",
#       response_format={ "type": "json_object" },
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant designed to output JSON. You are an expert in software engineer."},
#         {"role": "user", "content": f"I have a list of tags related to software engineer and computer science: {community_cores_with_cut1_sorted[ci]}. The tags are ordered by their importance and the fronter the more important. Please using no more than 30 words, give me a general task related to the following tags with 'task_description' as key name. Please make sure that the task is a general task and not too specific. Then try to conclude this task using less than 10 words with 'task_conclude' as key name. Please only return the json content and not anything else."}
#       ]
#     )
    
#     task_unweighted_level1_general[ci] = response.choices[0].message.content

# save_obj(task_unweighted_level1_general, f'gpt4_task_unweighted_level{level}_general', data_path_save + 'networks/probability/')


# from openai import OpenAI
# from tqdm import tqdm

# task_unweighted_level1_general_20word = {}
# ## input your key
# client = OpenAI(api_key='sk-proj-o7Whor')

# for ci in tqdm(community_cores_with_cut1_sorted.keys()):
    
#     response = client.chat.completions.create(
#       model="gpt-4-0125-preview",
#       response_format={ "type": "json_object" },
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant designed to output JSON. You are an expert in software engineer."},
#         {"role": "user", "content": f"I have a list of tags related to software engineer and computer science: {community_cores_with_cut1_sorted[ci]}. The tags are ordered by their importance and the fronter the more important. Please using no more than 20 words, give me a general task related to the following tags with 'task_description' as key name. Please make sure that the task is a general task and not too specific. Please only return the json content and not anything else."}
#       ]
#     )
    
#     task_unweighted_level1_general_20word[ci] = response.choices[0].message.content

# save_obj(task_unweighted_level1_general_20word, f'gpt4_task_unweighted_level{level}_general_20_words', data_path_save + 'networks/probability/')

# from openai import OpenAI
# from tqdm import tqdm

# task_unweighted_level1_general_10word = {}
# ## input your key
# client = OpenAI(api_key='sk-proj-o')

# for ci in tqdm(community_cores_with_cut1_sorted.keys()):
    
#     response = client.chat.completions.create(
#       model="gpt-4-0125-preview",
#       response_format={ "type": "json_object" },
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant designed to output JSON. You are an expert in software engineer."},
#         {"role": "user", "content": f"I have a list of tags related to software engineer and computer science: {community_cores_with_cut1_sorted[ci]}. The tags are ordered by their importance and the fronter the more important. Please using no more than 10 words, give me a general task related to the following tags with 'task_description' as key name. Please make sure that the task is a general task and not too specific. Please only return the json content and not anything else."}
#       ]
#     )
    
#     task_unweighted_level1_general_10word[ci] = response.choices[0].message.content

# save_obj(task_unweighted_level1_general_10word, f'gpt4_task_unweighted_level{level}_general_10_words', data_path_save + 'networks/probability/')

# from openai import OpenAI
# from tqdm import tqdm

# task_unweighted_level1_general_5word = {}

# ## input your key
# client = OpenAI(api_key='sk-proj')

# for ci in tqdm(community_cores_with_cut1_sorted.keys()):
    
#     response = client.chat.completions.create(
#       model="gpt-4-0125-preview",
#       response_format={ "type": "json_object" },
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant designed to output JSON. You are an expert in software engineer."},
#         {"role": "user", "content": f"I have a list of tags related to software engineer and computer science: {community_cores_with_cut1_sorted[ci]}. The tags are ordered by their importance and the fronter the more important. Please using no more than 5 words, give me a general task related to the following tags with 'task_description' as key name. Please make sure that the task is a general task and not too specific. Please only return the json content and not anything else."}
#       ]
#     )
    
#     task_unweighted_level1_general_5word[ci] = response.choices[0].message.content

# save_obj(task_unweighted_level1_general_5word, f'gpt4_task_unweighted_level{level}_general_5_words', data_path_save + 'networks/probability/')

###########################################################################################
from pickle_file import load_obj
import json
import pandas as pd

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {tc[0]:tc[1] for tc in tag_count}

for level in [1]:
    community_cores_with_cut = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')

    community_cores_with_cut_sorted = {c:[] for c in community_cores_with_cut.keys()}
    for c, ts in community_cores_with_cut.items():
        tsc = [tag_count_dict[t] for t in ts]
        if len(tsc) > 0:
            tsc, ts = zip(*sorted(zip(tsc,ts), reverse=True))
            community_cores_with_cut_sorted[c] = [t for t in ts]

    task_unweighted_level_general = load_obj(f'gpt4_task_unweighted_level{level}_general', data_path_save + 'networks/probability/')
    task_unweighted_level_general_20word = load_obj(f'gpt4_task_unweighted_level{level}_general_20_words', data_path_save + 'networks/probability/')
    task_unweighted_level_general_10word = load_obj(f'gpt4_task_unweighted_level{level}_general_10_words', data_path_save + 'networks/probability/')
    task_unweighted_level_general_5word = load_obj(f'gpt4_task_unweighted_level{level}_general_5_words', data_path_save + 'networks/probability/')
    
    list_c = []
    list_ts = []
    list_30 = []
    list_20 = []
    list_10 = []
    list_5 = []
    list_conclude = []
    for c, ts in community_cores_with_cut_sorted.items():
        list_c.append(c)
        list_ts.append(ts)
        if len(ts) > 0:
            if c == '231':
                d = json.loads(task_unweighted_level_general[c])
                list_30.append(d['task_description'])
                list_conclude.append(d['task_conclude'].replace(';',','))
                list_20.append(json.loads(task_unweighted_level_general_20word[c])['task_description'])
                list_10.append(json.loads(task_unweighted_level_general_10word[c])['task_description'])
                list_5.append(json.loads(task_unweighted_level_general_5word[c])['task_description'])
            else:
                d = json.loads(task_unweighted_level_general[c])
                list_30.append(d['task_description'])
                list_conclude.append(d['task_conclude'])
                list_20.append(json.loads(task_unweighted_level_general_20word[c])['task_description'])
                list_10.append(json.loads(task_unweighted_level_general_10word[c])['task_description'])
                list_5.append(json.loads(task_unweighted_level_general_5word[c])['task_description'])

        else:
            list_30.append(None)
            list_conclude.append(None)
            list_20.append(None)
            list_10.append(None)
            list_5.append(None)

    df = pd.DataFrame.from_dict({'task_id':list_c, 'tags':list_ts, 'task_description_30_words': list_30, 'task_description_20_words': list_20, 'task_description_10_words':list_10, 'task_description_5_words':list_5, 'task_description_conclude': list_conclude})

    df.to_csv(data_path_save + 'networks/probability/' + f'task_description_30_20_10_5_community_core_with_cut_level_{level}.csv')


###########################################################################################
##! some user location data
###########################################################################################
from pickle_file import load_obj, save_obj
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

df_user_location = pd.read_csv(data_path + 'user_date_loc_bing_fua.csv')

user_efua_by_year = {str(yr):defaultdict(str) for yr in range(2008,2024)}
user_country_by_year = {str(yr):defaultdict(str) for yr in range(2008,2024)}
yr_count = defaultdict(int)

for u_id, t, efua, country in tqdm(zip(df_user_location['user_id'], df_user_location['date_observed'], df_user_location['eFUA_name'], df_user_location['country'])):
    if not pd.isna(t):
        yr_count[t[:4]] += 1
        
    if not pd.isna(u_id) and not pd.isna(t) and not pd.isna(efua):
        user_efua_by_year[t[:4]][str(u_id)] = efua
        
    if not pd.isna(u_id) and not pd.isna(t) and not pd.isna(country):
        user_country_by_year[t[:4]][str(u_id)] = country.lower()

for yr in range(2008,2024):
    print(yr, len(user_efua_by_year[str(yr)]), len(user_country_by_year[str(yr)]), yr_count[str(yr)])
    save_obj(user_efua_by_year[str(yr)], f'user_efua_{yr}', data_path_save + 'user_location/')
    save_obj(user_country_by_year[str(yr)], f'user_country_{yr}', data_path_save + 'user_location/')

from pickle_file import load_obj, save_obj
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

year_period_list = [[i for i in range(2009,2016)],[i for i in range(2011,2016)],[i for i in range(2016,2021)],[i for i in range(2021,2024)]]


for yr in range(2008,2024):
    user_efua_by_year = load_obj(f'user_efua_{yr}', data_path_save + 'user_location/')
    user_country_by_year = load_obj(f'user_country_{yr}', data_path_save + 'user_location/')
        
    efua_users_by_year = defaultdict(list)
    for u, efua in user_efua_by_year.items():
        efua_users_by_year[efua].append(u)

    country_users_by_year = defaultdict(list)
    for u, efua in user_country_by_year.items():
        country_users_by_year[efua].append(u)

    print(len(efua_users_by_year), len(country_users_by_year))

    save_obj(efua_users_by_year, f'efua_users_{yr}',data_path_save + 'user_location/')
    save_obj(country_users_by_year, f'country_users_{yr}',data_path_save + 'user_location/')

###########################################################################################
##! build all the task data
###########################################################################################
import importlib
import answer_vote_regression_did #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_regression_did)
from answer_vote_regression_did import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

programming_language_std_adjusted = set(load_obj('programming_language_std_adjusted', data_path_save))

level_list = [1,3,2]
year_list = [yr for yr in range(2008,2024)]

##! user的所有tag，包括没有用来建立网络的
print('set_user_tagall_set')
get_user_answer_tagall_set(data_path, data_path_save)


for level in level_list:
    print("level  ", level)
    tag_bool_core = load_obj(f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')
    G_tag_core = load_obj(f"G_tag_core_with_cut_level{level}", data_path_save + 'networks/probability/')

    #* user的各种task数量, 'user_task_count_by_year_level_{level}_{yr}'
    print('get_user_task_dict')
    get_user_task_dict(data_path, data_path_save, G_tag_core, tag_bool_core, year_list, level)
    #* user的task set
    get_answer_task_set(data_path_save, tag_bool_core, year_list, level)

    #* answer的language数量和task数量, 'answer_language_length_{level}_{yr}', 'answer_task_length_{level}_{yr}'
    print('get_answer_task_language_length')
    get_answer_task_language_length(data_path, data_path_save, programming_language_std_adjusted, G_tag_core, tag_bool_core, level)

    get_user_task_language_dict(data_path, data_path_save, programming_language_std_adjusted, year_list, G_tag_core, tag_bool_core, level)

    
#* user的各种language数量, 'user_language_count_by_year_{yr}'
print('get_user_language_dict')
get_user_language_dict(data_path, data_path_save, programming_language_std_adjusted, year_list)

#* [0: parent_answer_num, 1: parent_answer_vote, 2: parent_vote, 3: parent_id]
#* 'answer_parent_num_vote_{yr}'
print('get_answer_parent_vote')
get_answer_parent_vote(data_path, data_path_save)

#* user的answer history, 'user_answer_history_{yr}'
print('get_user_answer_history')
get_user_answer_history(data_path, data_path_save)

print('get_python_user')
get_python_user(data_path, data_path_save, 0.1)

print('get_question_answer_time')
get_question_answer_time(data_path, data_path_save)

print('get_answer_ranks')
get_answer_vote_rank(data_path, data_path_save)

print('get_answer_time_ranks')
get_answer_time_rank(data_path, data_path_save)

###########################################################################################
import importlib
import answer_vote_regression_did #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_regression_did)
from answer_vote_regression_did import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


#* user d major language
level = 1

year_period = 2

for yr in range(2008 + year_period + 1, 2024):
    yr_list = [yr - year_period + i for i in range(year_period)]
    print(yr_list)
    collect_user_major_language_year(data_path_save, yr_list)


for yr in range(2008, 2024):
    yr_list = [yr]
    print(yr_list)
    collect_user_major_language_year(data_path_save, yr_list)

collect_user_major_language_year(data_path_save, [yr for yr in range(2008, 2024)])

###########################################################################################
##! density matrix
###########################################################################################
from pickle_file import load_obj, save_obj
import importlib
import task_prediction #import the module here, so that it can be reloaded.
importlib.reload(task_prediction)
from task_prediction import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


##! label
level = 1
sample_percent = 0.5
sample_threshold = 10

##! build community colocation matrix
year_bool = {str(yr):False for yr in range(2008, 2024)}
for yr in range(2008,2024):
    year_bool[str(yr)] = True

print('================================= user sample =================================')
#sample_half_user_bool, threshold_user_bool = sample_from_users(year_bool, data_path_save, sample_percent, sample_threshold)

##! sample_half_user_bool: 至少10个answer的user中的一半
##! threshold_user_bool: 至少10个answer的user
#save_obj(sample_half_user_bool, 'half_user_bool', data_path_save + f'vote_regression_together/')
#save_obj(threshold_user_bool, 'all_threshold_user_bool', data_path_save + f'vote_regression_together/')

##! all user bool
#all_answer_user_bool = get_all_user_bool(data_path_save, 'answer')
#save_obj(all_answer_user_bool, 'all_answer_user_bool', data_path_save + f'vote_regression_together/')

###########################################################################################
import importlib
import task_prediction #import the module here, so that it can be reloaded.
importlib.reload(task_prediction)
from task_prediction import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'



##! label
user_bool = load_obj('half_user_bool', data_path_save + f'vote_regression_together/')

for level in [1,2,3]:

    ##! load network and community
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

    ##! build community colocation matrix
    print('================================= all years user community vector =================================')
    year_bool = {str(yr):False for yr in range(2008, 2024)}
    for yr in range(2008,2024):
        year_bool[str(yr)] = True

    user_community_set,  user_community_vector_binary = get_user_community_set_from_sample(year_bool, user_bool, data_path_save, community_list_core_std, level)
    cc_pmi, cc_pmi_matrix, Q = build_community_cooccurrence(user_community_set, community_list_core_std, 1/len(community_list_core_std))
    save_obj(cc_pmi, f'cc_pmi_half_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    save_obj(cc_pmi_matrix, f'cc_pmi_matrix_normalized_half_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    del user_community_set
    del user_community_vector_binary


    ##! 2008 - 2012, build community colocation matrix
    print('=================================2008 to 2012 user community vector =================================')
    year_bool = {str(yr):False for yr in range(2008, 2024)}
    for yr in range(2008,2013):
        year_bool[str(yr)] = True

    user_community_set,  user_community_vector_binary = get_user_community_set_from_sample(year_bool, user_bool, data_path_save, community_list_core_std, level)
    cc_pmi, cc_pmi_matrix, Q = build_community_cooccurrence(user_community_set, community_list_core_std, 1/len(community_list_core_std))
    save_obj(cc_pmi, f'cc_pmi_half_user_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    save_obj(cc_pmi_matrix, f'cc_pmi_matrix_normalized_half_user_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    del user_community_set
    del user_community_vector_binary
###########################################################################################

import importlib
import task_prediction #import the module here, so that it can be reloaded.
importlib.reload(task_prediction)
from task_prediction import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'



##! label
user_bool = load_obj('all_threshold_user_bool', data_path_save + f'vote_regression_together/')

for level in [1,2,3]:

    ##! load network and community
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    
    ##! build community colocation matrix
    print('================================= all years user community vector =================================')
    year_bool = {str(yr):False for yr in range(2008, 2024)}
    for yr in range(2008,2024):
        year_bool[str(yr)] = True

    user_community_set,  user_community_vector_binary = get_user_community_set_from_sample(year_bool, user_bool, data_path_save, community_list_core_std, level)
    cc_pmi, cc_pmi_matrix, Q = build_community_cooccurrence(user_community_set, community_list_core_std, 1/len(community_list_core_std))
    save_obj(cc_pmi, f'cc_pmi_all_threshold_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    save_obj(cc_pmi_matrix, f'cc_pmi_matrix_normalized_all_threshold_user_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    del user_community_set
    del user_community_vector_binary


    ##! 2008 - 2012, build community colocation matrix
    print('=================================2008 to 2012 user community vector =================================')
    year_bool = {str(yr):False for yr in range(2008, 2024)}
    for yr in range(2008,2013):
        year_bool[str(yr)] = True

    user_community_set,  user_community_vector_binary = get_user_community_set_from_sample(year_bool, user_bool, data_path_save, community_list_core_std, level)
    cc_pmi, cc_pmi_matrix, Q = build_community_cooccurrence(user_community_set, community_list_core_std, 1/len(community_list_core_std))
    save_obj(cc_pmi, f'cc_pmi_all_threshold_user_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    save_obj(cc_pmi_matrix, f'cc_pmi_matrix_normalized_all_threshold_user_2008_2012_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')
    del user_community_set
    del user_community_vector_binary

###########################################################################################
##! task language matrix
###########################################################################################
import importlib
import task_language_matrix #import the module here, so that it can be reloaded.
importlib.reload(task_language_matrix)
from task_language_matrix import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


threshold_user_bool = load_obj('all_threshold_user_bool', data_path_save + f'vote_regression_together/')


language_merge_dict = {'python':['python_3.x', 'python_2.7', 'python_3.6', 'python_2.x', 'python_3.4', 'python_3.3', 'python_3.10', 'python_2.5'],
                    'c++':['c++11', 'c++17', 'c++14'],
                    'actionscript_3':['actionscript', 'actionscript_2'],
                    'asp.net_mvc':['asp.net_mvc_3', 'asp.net_mvc_4'],
                    'sql_server':['sql_server_2008', 'sql_server_2005'],
                    'laravel':['laravel_5'],
                    'c#':['c#_4.0', 'c#_2.0', 'c#_5.0', 'c#_8.0'],
                    'swift':['swift3', 'swift2','swift4','swift4.2'],
                    'java':['java_8'],
                    'ruby':['ruby_2.0'],
                    'ruby_on_rails':['ruby_on_rails_3', 'ruby_on_rails_4', 'ruby_on_rails_7'],
                    'powershell':['powershell_3.0', 'powershell_4.0', 'powershell_5.0']}


for level in [1]:

    programming_language_std_adjusted = load_obj('programming_language_std_adjusted', data_path_save)
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

    for yr in range(2008, 2024):
        print(yr)
        #collect_task_language_user_count_matrix_answer(data_path_save, [yr], level, programming_language_std_adjusted, community_list_core_std, threshold_user_bool)
        collect_task_language_user_count_matrix_answer_with_merge(data_path_save, [yr], level, programming_language_std_adjusted, community_list_core_std, threshold_user_bool, language_merge_dict)
        collect_language_user_count_matrix_answer_with_merge(data_path_save, [yr], programming_language_std_adjusted, threshold_user_bool, language_merge_dict)

###########################################################################################

import importlib
import task_language_matrix #import the module here, so that it can be reloaded.
importlib.reload(task_language_matrix)
from task_language_matrix import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


language_merge_dict = {'python':['python_3.x', 'python_2.7', 'python_3.6', 'python_2.x', 'python_3.4', 'python_3.3', 'python_3.10', 'python_2.5'],
                    'c++':['c++11', 'c++17', 'c++14'],
                    'actionscript_3':['actionscript', 'actionscript_2'],
                    'asp.net_mvc':['asp.net_mvc_3', 'asp.net_mvc_4'],
                    'sql_server':['sql_server_2008', 'sql_server_2005'],
                    'laravel':['laravel_5'],
                    'c#':['c#_4.0', 'c#_2.0', 'c#_5.0', 'c#_8.0'],
                    'swift':['swift3', 'swift2','swift4','swift4.2'],
                    'java':['java_8'],
                    'ruby':['ruby_2.0'],
                    'ruby_on_rails':['ruby_on_rails_3', 'ruby_on_rails_4', 'ruby_on_rails_7'],
                    'powershell':['powershell_3.0', 'powershell_4.0', 'powershell_5.0']}

threshold_user_bool = load_obj('all_threshold_user_bool', data_path_save + f'vote_regression_together/')

for level in [1]:

    programming_language_std_adjusted = load_obj('programming_language_std_adjusted', data_path_save)
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

    #collect_task_language_user_count_matrix_answer(data_path_save, [yr for yr in range(2008, 2024)], level, programming_language_std_adjusted, community_list_core_std, threshold_user_bool)
    collect_task_language_user_count_matrix_answer_with_merge_all_year(data_path_save, [yr for yr in range(2008, 2024)], level, programming_language_std_adjusted, community_list_core_std, threshold_user_bool, language_merge_dict)
    collect_language_user_count_matrix_answer_with_merge_all_year(data_path_save, [yr for yr in range(2008, 2024)], programming_language_std_adjusted, threshold_user_bool, language_merge_dict)
    
###########################################################################################
###########################################################################################
##! task values
###########################################################################################

import importlib
import task_value_from_survey #import the module here, so that it can be reloaded.
importlib.reload(task_value_from_survey)
from task_value_from_survey import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

country_dict = load_obj('country_dict', data_path_save+f'surveys/country/')
country_count = load_obj('country_count', data_path_save+f'surveys/country/')

country_dict_temp = {c:cs for c,cs in country_dict.items() if cs == 'United States'}
country_dict = country_dict_temp

country_count_threshold = 10

period_label = 'hn_job_task_salary_only_us_log'
get_sv_salary_log(data_path, data_path_save, country_count, country_dict, country_count_threshold, period_label)

get_sv_tags(data_path, data_path_save, period_label)
get_sv_salary_tags(data_path, data_path_save, period_label)

###########################################################################################
import importlib
import task_value_from_survey #import the module here, so that it can be reloaded.
importlib.reload(task_value_from_survey)
from task_value_from_survey import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

period_label = 'hn_job_task_salary_only_us_log'

so_yearlist =  [2018, 2019,2020, 2021, 2022, 2023]
sv_yearlist = [2023]

topn = 300
print('topn: ', topn, so_yearlist, sv_yearlist)

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

product_so_tags_sv_topn(so_yearlist, sv_yearlist, period_label, data_path_save, topn, sample_user_label)
task_salary_topn = {}
for level in [3,2,1]:
    task_salary_topn[level] = calculate_task_salary_topn(data_path_save, period_label, so_yearlist, sv_yearlist, level, topn, sample_user_label)



from pickle_file import load_obj, save_obj

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

period_label = 'hn_job_task_salary_only_us_log'

so_yearlist =  [2018, 2019,2020, 2021, 2022, 2023]
sv_yearlist = [2023]

topn = 300
print('topn: ', topn, so_yearlist, sv_yearlist)
level =1

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

task_salary = load_obj(f'{so_yearlist}_{sv_yearlist}_task_salary_topn_{topn}_level_{level}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')
save_obj(task_salary, f'df_task_value_2023', data_path_save + f'surveys/country/{period_label}/')

###########################################################################################
##! python-all language, build dataframe
###########################################################################################
import importlib
import user_task_collection #import the module here, so that it can be reloaded.
importlib.reload(user_task_collection)
from user_task_collection import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

programming_language_std_adjusted = set(load_obj('programming_language_std_adjusted', data_path_save))

level_list = [1]
year_list = [yr for yr in range(2008,2024)]

for level in level_list:
    print("level  ", level)
    tag_bool_core = load_obj(f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')
    G_tag_core = load_obj(f"G_tag_core_with_cut_level{level}", data_path_save + 'networks/probability/')

    #* user的各种task数量, 'user_task_count_by_year_level_{level}_{yr}'
    print('get_user_task_dict')
    get_user_task_dict_by_language(data_path, data_path_save, G_tag_core, tag_bool_core, year_list, level, programming_language_std_adjusted, 'python')
###########################################################################################
import importlib
import user_task_collection #import the module here, so that it can be reloaded.
importlib.reload(user_task_collection)
from user_task_collection import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

level_list = [1]
year_list = [yr for yr in range(2009,2023)]
year_list.reverse()
for level in level_list:

    get_user_starting_year(data_path_save, year_list)

    get_user_starting_year_by_language(data_path_save,year_list, 'python')
###########################################################################################
import importlib
import user_task_collection #import the module here, so that it can be reloaded.
importlib.reload(user_task_collection)
from user_task_collection import *
random.seed(1730)

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

level = 1

community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

year_period = 1
#density_user_label = 'half_user'
density_user_label = 'all_threshold_user'

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

so_yearlist_salary = [2018, 2019, 2020, 2021, 2022, 2023]
sv_yearlist_salary = [2023]
topn = 300
period_label = 'hn_job_task_salary_only_us_log'

task_salary_name = [f'{so_yearlist_salary}_{sv_yearlist_salary}_task_salary_topn_{topn}_level_{level}', data_path_save + f'surveys/country/{period_label}/']

#sample_percent_label = 1
sample_percent_label = 10

# year_bool = {str(yr):False for yr in range(2008, 2024)}
# for yr in range(2008,2024):
#     year_bool[str(yr)] = True

print('================================= user sample =================================')
# user_pool_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
# user_list = list(set([u for u, ub in user_pool_bool.items() if ub]))
# user_sample = random.sample(user_list, int(len(user_list) * sample_percent_label/100))
# selected_user_bool = defaultdict(bool)
# for u in user_sample:
#     selected_user_bool[u] = True

selected_user_bool = load_obj('user_selection_bool_10percent', data_path_save + 'vote_regression_together/user_task_collection/df_sample/')


for yr1 in range(2009, 2023 - year_period):
    print('================================= build dataframe all language =================================')
    yr_list1 = [yr1]
    print(yr_list1)
    yr2 = yr1 + year_period
    yr_list2 = [yr2]
    build_user_task_entry_exit_salary_dataframe_with_sample_all_language(data_path_save, community_list_core_std, yr_list1, yr1, yr_list2, yr2, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label)

dataframe_names = [data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_from_{yr1}_to_{yr1 + 1}_level_{level}_{sample_percent_label}_percent_all_language.csv' for yr1 in range(2009, 2023 - year_period)]

save_name = data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_level_{level}_{sample_percent_label}_percent_all_language.csv'

df_all = pd.concat([pd.read_csv(dfn) for dfn in dataframe_names])
df_all.to_csv(save_name)


for yr1 in range(2009, 2023 - year_period):
    print('================================= build dataframe python =================================')
    yr_list1 = [yr1]
    print(yr_list1)
    yr2 = yr1 + year_period
    yr_list2 = [yr2]
    build_user_task_entry_exit_salary_dataframe_with_sample_python(data_path_save, community_list_core_std, yr_list1, yr1, yr_list2, yr2, level, selected_user_bool, density_user_label, sample_user_label, sample_percent_label)


dataframe_names = [data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_from_{yr1}_to_{yr1 + 1}_level_{level}_{sample_percent_label}_percent_python.csv' for yr1 in range(2009, 2023 - year_period)]

save_name = data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_level_{level}_{sample_percent_label}_percent_python.csv'

df_all = pd.concat([pd.read_csv(dfn) for dfn in dataframe_names])
df_all.to_csv(save_name)

###########################################################################################
import importlib
import user_task_collection #import the module here, so that it can be reloaded.
importlib.reload(user_task_collection)
from user_task_collection import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


level = 1
community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

#sample_user_label = 'half_user'
#sample_user_label = 'all_threshold_user'
sample_user_label = 'all_answer_user'

for yr in range(2008, 2024):
    yr_list = [yr]
    print(yr_list)
    build_user_activity_dataframe(data_path_save, yr_list, yr, level, sample_user_label)

###########################################################################################
df_list = []
for yr in range(2008, 2024):
    year_label = yr
    df = pd.read_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_activity_{year_label}_level_{level}.csv')
    df_list.append(df)

df_all = pd.concat(df_list)
df_all.to_csv(data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_{sample_user_label}_task_activity_2008_2023_level_{level}.csv')
###########################################################################################
###########################################################################################
##! answer-vote, build dataframe
###########################################################################################
import importlib
import answer_vote_regression_did #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_regression_did)
from answer_vote_regression_did import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


programming_language_std_adjusted = set(load_obj('programming_language_std_adjusted', data_path_save))

level_list = [1]
year_list = [yr for yr in range(2008,2024)]

experience_period_length = 2

for level in level_list:
    tag_bool_core = load_obj(f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')
    G_tag_core = load_obj(f"G_tag_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    collect_answer_vote_coefficients(data_path, data_path_save, level, programming_language_std_adjusted, G_tag_core,tag_bool_core, experience_period_length, language_len = [1], task_len = [1,2,3,4,5,6,7])
###########################################################################################
import importlib
import answer_vote_regression_did #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_regression_did)
from answer_vote_regression_did import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'



experience_period_length = 2
level = 1

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

regression_year_list = [i for i in range(2011, 2024)]
df = vote_regression_build_dataframe(data_path_save, regression_year_list, experience_period_length, level, sample_user_label)
df.to_csv(data_path_save + f'vote_regression_together/vote_regression_dataframe_{sample_user_label}_{regression_year_list[0]}_{regression_year_list[-1]}_period_{experience_period_length}_level_{level}.csv')
del df


###########################################################################################
import importlib
import answer_vote_regression_did_parallel #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_regression_did_parallel)
from answer_vote_regression_did_parallel import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


experience_period_length = 2

level = 1


# tuple 的list
input_list = [yr for yr in range(2008 + experience_period_length + 1, 2024)]

from joblib import Parallel, delayed
Parallel(n_jobs=16)(delayed(collect_answer_task_minute_experience_build_dataframe_parallel)(yr) for yr in input_list)



###########################################################################################
import importlib
import answer_vote_task_IV #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_task_IV)
from answer_vote_task_IV import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'



experience_period_length = 2
level = 1

# sample_user_label = 'all_threshold_user'
# get_answer_vote_rank_confine_user(data_path, data_path_save, sample_user_label)

# sample_user_label = 'all_answer_user'
# get_answer_vote_rank_confine_user(data_path, data_path_save, sample_user_label)

get_answer_user_dict(data_path, data_path_save)
###########################################################################################
import importlib
import answer_vote_task_IV #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_task_IV)
from answer_vote_task_IV import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'



experience_period_length = 2
level = 1

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

regression_year_list = [i for i in range(2011, 2024)]
# for yr in regression_year_list:
#     get_yr_task_abs_minute_exp(data_path_save, level, experience_period_length, yr, sample_user_label)

# tuple 的list
#input_list = [(data_path_save, level, experience_period_length, yr, sample_user_label) for yr in regression_year_list]
# from joblib import Parallel, delayed
# Parallel(n_jobs=16)(delayed(get_yr_task_abs_minute_exp)(*ipt) for ipt in input_list)

# for yr in regression_year_list:
#     get_yr_history_abs_minute_exp(data_path_save, experience_period_length, yr, sample_user_label)

input_list = [(data_path_save, experience_period_length, yr, sample_user_label) for yr in regression_year_list]
from joblib import Parallel, delayed
Parallel(n_jobs=16)(delayed(get_yr_history_abs_minute_exp)(*ipt) for ipt in input_list)

###########################################################################################
import importlib
import answer_vote_task_IV #import the module here, so that it can be reloaded.
importlib.reload(answer_vote_task_IV)
from answer_vote_task_IV import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


# sample_user_label = 'all_threshold_user'
# get_answer_vote_rank_confine_user(data_path, data_path_save, sample_user_label)

sample_user_label = 'all_answer_user'
get_answer_vote_rank_confine_user(data_path, data_path_save, sample_user_label)

experience_period_length = 2
level = 1

#sample_user_label = 'half_user'
sample_user_label = 'all_threshold_user'
#sample_user_label = 'all_answer_user'

regression_year_list = [i for i in range(2011, 2024)]

# tuple 的list
input_list = [(data_path_save, experience_period_length, level, sample_user_label, yr) for yr in regression_year_list]

from joblib import Parallel, delayed
Parallel(n_jobs=8)(delayed(vote_task_regression_IV_json)(*ipt) for ipt in input_list)

vote_task_exp_IV_build_dataframe(data_path_save, experience_period_length, level, sample_user_label, regression_year_list)
###########################################################################################
###########################################################################################
##! task space, node position
###########################################################################################
import importlib
import draw_task_space #import the module here, so that it can be reloaded.
importlib.reload(draw_task_space)
from draw_task_space import *

import umap
import hdbscan

##! get the task-task density, by task cooccurrence in users from 2008 to 2023. The users are ones with at least 10 answers in all time.

level = 1

df,min_rca_t = get_task_task_density_no_empty(level)

min_rca_t_coefficient = 2


community_list_std1 =  load_obj(f"community_list_std_core_cut_level1", data_path_save + 'networks/probability/')
community_unweighted_level1 = load_obj(f"community_core_with_cut_level1", data_path_save + 'networks/probability/')
community_list_std1_no_empty = [c for c in community_list_std1 if len(community_unweighted_level1[c] ) > 0]
community_list_std2 =  load_obj(f"community_list_std_core_cut_level2", data_path_save + 'networks/probability/')
community_unweighted_level2 = load_obj(f"community_core_with_cut_level2", data_path_save + 'networks/probability/')
community_list_std3 =  load_obj(f"community_list_std_core_cut_level3", data_path_save + 'networks/probability/')
community_unweighted_level3 = load_obj(f"community_core_with_cut_level3", data_path_save + 'networks/probability/')
C_dict3 = {c:i for i,c in enumerate(community_list_std3)}

metro_colors = ['#EA0437', '#87D300', '#FFD100', '#4F1F91', '#A24CC8', '#FF7200', '#009EDB', '#78C7EB', '#BC87E6', '#7C2230', '#007B63', '#D71671', '#F293D1', '#7F7800', '#BBA786', '#32D4CB', '#B67770', '#D6A461', '#DFC765', '#666666', '#999999', '#009090', '#EE352E', '#00933C', '#B933AD', '#808183', '#0039A6', '#FF6319', '#6CBE45', '#996633', '#A7A9AC', '#FCCC0A', '#00ADD0', '#00985F', '#60269E', '#4D5357', '#6E3219', '#CE8E00', '#006983', '#00AF3F', '#C60C30', '#A626AA', '#00A1DE', '#009B3A', '#EE0034', '#8E258D', '#FF7900', '#6E267B', '#A4343A', '#004B87', '#D90627', '#008C95', '#AA0061', '#B58500', '#FFC56E', '#009B77', '#97D700', '#0092BC', '#FF8674', '#9C4F01', '#F4DA40', '#CA9A8E', '#653279', '#6BA539', '#00ABAB', '#D3A3C9', '#F4C1CA', '#D0006F', '#D86018', '#A45A2A', '#D986BA', '#476205', '#D22630', '#A192B2', '#0049A5', '#FF9500', '#F62E36', '#B5B5AC', '#009BBF', '#00BB85', '#C1A470', '#8F76D6', '#00AC9B', '#9C5E31', '#003DA5', '#77C4A3', '#F5A200', '#0C8E72', '#204080', '#C30E2F', '#1CAE4C', '#5288F5', '#E06040', '#3D99C2', '#80E080', '#3D860B', '#3698D2', '#074286', '#1D2A56', '#753778', '#F9BE00', '#2B3990', '#0052A4', '#009D3E', '#EF7C1C', '#00A5DE', '#996CAC', '#CD7C2F', '#747F00', '#EA545D', '#A17E46', '#BDB092', '#B7C452', '#B0CE18', '#0852A0', '#6789CA', '#941E34', '#B21935', '#8A5782', '#9A6292', '#59A532', '#7CA8D5', '#ED8B00', '#FFCD12', '#22246F', '#FDA600', '#0065B3', '#0090D2', '#F97600', '#6FB245', '#509F22', '#ED1C24', '#D4003B', '#8EC31F', '#81A914', '#E04434', '#F04938', '#A17800', '#C7197D', '#5A2149', '#2A5CAA', '#F06A00', '#81BF48', '#BB8C00', '#217DCB', '#D6406A', '#68BD45', '#8652A1', '#004EA2', '#D93F5C', '#00AA80', '#FFB100', '#F08200', '#009088', '#009362', '#007448', '#FF0000', '#009900']

community_hierarchy_color = [] 
for c in community_list_std1_no_empty:
    for ch, tsh in community_unweighted_level3.items():
        if len(set(community_unweighted_level1[c]).intersection(set(tsh)))>0:
            community_hierarchy_color.append(metro_colors[C_dict3[ch]])
            break
        
    if c=='214':
        community_hierarchy_color.append('#B2BEB5')

occ_colors=pd.DataFrame(['community_'+c for c in community_list_std1_no_empty],columns=['occ_code'])
occ_colors['color']=community_hierarchy_color

occspace = df
print(occspace.shape)

print(occspace.index.equals(occspace.columns))

occspace = occspace.fillna(0)


occspace = 1/occspace
occspace = occspace.replace([np.inf, -np.inf], np.nan)
occspace = occspace.fillna(1/min_rca_t**min_rca_t_coefficient)
distmat = occspace.values

np.fill_diagonal(distmat,0)

df.to_csv(data_path_save + 'task_space_draw/df_task_density_1.csv')

import matplotlib.pyplot as plt

def umapnn(nn=5,md=0.1,n_components=2):
    # init, set metric as "precomputed", and two important parameter.
    # Check https://umap-learn.readthedocs.io/en/latest/parameters.html for parameter explanations.
    reducer = umap.UMAP(metric='precomputed',n_neighbors=nn,random_state=42,min_dist=md,n_components=2)
    position = reducer.fit_transform(distmat)
    # the following just create a df to hold result and merge other properties for viz.
    umapdf = pd.DataFrame(position)
    umapdf.columns=['x','y']
    umapdf['occ_code'] = occspace.index
    umapdf = umapdf.merge(occ_colors[['occ_code','color']],how='left')
    #umapdf = umapdf.merge(occ_labels[['occ_code','label']],how='left')
    plt.figure(figsize=(8,6))
    plt.scatter(umapdf.x,umapdf.y,c=umapdf.color)
    #plt.scatter(umapdf.x,umapdf.y)
    return(umapdf)

def umapDBSCAN(nn=10,md=0.001,n_components=5):
#def umapDBSCAN(nn=15,md=0.1,n_components=30):
    # init, set metric as "precomputed", and two important parameter.
    # Check https://umap-learn.readthedocs.io/en/latest/parameters.html for parameter explanations.
    reducer = umap.UMAP(metric='precomputed',n_neighbors=nn,random_state=42,min_dist=md,n_components=n_components)
    position = reducer.fit_transform(distmat)
    # the following just create a df to hold result and merge other properties for viz.
    umapdf = pd.DataFrame(position)
    umapdf['occ_code'] = occspace.index
    umapdf = umapdf.merge(occ_colors[['occ_code','color']],how='left')
    #umapdf = umapdf.merge(occ_labels[['occ_code','label']],how='left')
    return(umapdf)


umapdf=umapnn(50,0.01,2)
umapdf_DBSCAN=umapDBSCAN(20, 0.001, 5)
len(umapdf_DBSCAN)


import seaborn as sns

col = 0
test_data=umapdf_DBSCAN.drop(['occ_code','color'],axis=1).values
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True,min_samples=2,cluster_selection_epsilon=.3)
clusterer.fit(test_data)
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#palette = sns.color_palette(n_colors=len(set(clusterer.labels_)))
cluster_colors = [sns.desaturate(metro_colors[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
print('# not classified:', sum(clusterer.labels_==-1))
print('# classes:', len(set(clusterer.labels_)))
###########################################################################################
print(len(umapdf_DBSCAN))
umapdf_DBSCAN2=umapdf_DBSCAN.copy()
umapdf_DBSCAN2['cluster']=clusterer.labels_
umapdf_DBSCAN2['clust_prob']=clusterer.probabilities_
umapdf_DBSCAN2=umapdf_DBSCAN2.merge(umapdf[['occ_code','x','y']],how='right',on='occ_code').reset_index()

pd.set_option('display.max_rows', 500)
print(umapdf_DBSCAN2.cluster.nunique())

print()

umapdf_DBSCAN2['cluster_inc_all'] = umapdf_DBSCAN2['cluster']

umapdf_DBSCAN2['cluster_inc_all'] = umapdf_DBSCAN2['cluster']
UMAP_coor_cols = [i for i in umapdf_DBSCAN2.columns.values if not isinstance(i, str)]

av_coor_clust = umapdf_DBSCAN2[umapdf_DBSCAN2['cluster']!=-1][UMAP_coor_cols]
clust_list = umapdf_DBSCAN2[umapdf_DBSCAN2['cluster']!=-1]['cluster']
def find_closest_cluster(x,av_coor_clust,clust_list):
    dists = []
    for i in av_coor_clust.values:
        dists.append(np.linalg.norm((x.values-i)))
    loc = np.argmin(dists)    
    best_clust = clust_list.iloc[loc]
    return(best_clust)
umapdf_DBSCAN2.loc[umapdf_DBSCAN2['cluster_inc_all']==-1,'cluster_inc_all'] = umapdf_DBSCAN2.loc[umapdf_DBSCAN2['cluster_inc_all']==-1,UMAP_coor_cols].apply(find_closest_cluster, axis=1, args=(av_coor_clust,clust_list,))

plt.figure(figsize=(15,10))
plt.rcParams['font.family'] = ['sans-serif']


df_task_description = pd.read_csv(data_path_save + 'networks/probability/' + 'task_description_30_20_10_5_community_core_with_cut_level_1.csv', dtype=str)
df_dict_5 = pd.Series(df_task_description.task_description_5_words.values,index=df_task_description.task_id).to_dict()


cluster_colors_new = [sns.desaturate(metro_colors[col], 1)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(umapdf_DBSCAN2.cluster_inc_all, umapdf_DBSCAN2.index)]

umapdf_DBSCAN2['node_color'] = cluster_colors_new

task_size = get_task_size([yr for yr in range(2008, 2024)], 1)
ms_size_dict = {t[10:]:task_size[t[10:]] for t in umapdf_DBSCAN2.occ_code}
#ms_size = [np.log10(ms_size_dict[umapdf_DBSCAN2.occ_code[i][10:]]) * 30 - 40 for i in umapdf_DBSCAN2.index]
ms_size = [np.sqrt(ms_size_dict[umapdf_DBSCAN2.occ_code[i][10:]]) *0.5 + 10 for i in umapdf_DBSCAN2.index]

umapdf_DBSCAN2['node_size'] = ms_size

plt.scatter(umapdf_DBSCAN2.x, umapdf_DBSCAN2.y, c=umapdf_DBSCAN2.node_color, s = umapdf_DBSCAN2.node_size)


from adjustText import adjust_text
text_list = []

for i in umapdf_DBSCAN2.index:
    text_list.append(plt.text(umapdf_DBSCAN2.x[i], umapdf_DBSCAN2.y[i], umapdf_DBSCAN2.occ_code[i][10:], size = 8))

###########################################################################################
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
plt.rcParams['font.family'] = ['sans-serif']

# Create some example data
x = [umapdf_DBSCAN2.x[i] for i in umapdf_DBSCAN2.index]
y = [umapdf_DBSCAN2.y[i] for i in umapdf_DBSCAN2.index]
task_size_list_yr = defaultdict(list)


tl_periods = [[2008,2009,2010,2011], [2012,2013,2014,2015],[2016,2017,2018,2019],[2020,2021,2022,2023]]

for tlp in tl_periods:
    task_user_set_all = defaultdict(set)
    for yr in tlp:
        task_user_set = load_obj(f'task_user_set_{yr}_level_{level}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, tsl in task_user_set.items():
            for t in tsl:
                task_user_set_all[t].add(u)

    task_user_count = {t:len(ul) for t, ul in task_user_set_all.items()}
    task_size_list = [np.sqrt(task_user_count[t[10:]]) * 2.3 for t in umapdf_DBSCAN2.occ_code]
    task_size_normalized = [si/sum(task_size_list) for si in task_size_list]

    #tuned_min = 20
    #tuned_max = 500

    #a_linear = (tuned_max - tuned_min) /(max(task_size_normalized) - min(task_size_normalized))
    #b_linear = tuned_max - a_linear * max(task_size_normalized)
    #adjust_list = [a_linear * si + b_linear for si in task_size_normalized]
    adjust_list = task_size_list

    for s,t in zip(adjust_list, umapdf_DBSCAN2.occ_code):
        task_size_list_yr[t[10:]].append(s)

task_size_max = {t:max(sl) for t, sl in task_size_list_yr.items()}

###########################################################################################
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,10))
plt.rcParams['font.family'] = ['sans-serif']

# Create some example data
x = [umapdf_DBSCAN2.x[i] for i in umapdf_DBSCAN2.index]
y = [umapdf_DBSCAN2.y[i] for i in umapdf_DBSCAN2.index]
s = [task_size_max[umapdf_DBSCAN2.occ_code[i][10:]] for i in umapdf_DBSCAN2.index]
index_list = [i for i in range(len(x))]

iterate_sign = True
iterate_num = 0

while(iterate_sign):
    iterate_num += 1
    if iterate_num % 200 == 0:
        print(iterate_num)
    iterate_sign = False
    sc = plt.scatter(x, y, s=s)
    ax = plt.gca()
    R = []
    for i in range(len(x)):
        #data_units_xi, data_units_yi = points_to_data_units(fig, s[i]**0.5, ax)
        #R.append((np.sqrt(max(data_units_xi**2, data_units_yi**2) *1)))
        R.append(s[i]**0.5 * 1.)

    x1,y1 = points_to_data_units(fig, 1, ax)

    for i in index_list:
        for j in index_list:
            if i < j:
                temp = np.sqrt((x[i]/x1-x[j]/x1)**2 + (y[i]/y1-y[j]/y1)**2) - R[i] - R[j]
                if temp < 0:
                    #print(temp)
                    move_dist = np.abs(temp)/np.sqrt(2)
                    move_dist_x = np.sqrt((x[i]/x1-x[j]/x1)**2 / ((x[i]/x1-x[j]/x1)**2 + (y[i]/y1-y[j]/y1)**2)) * move_dist
                    move_dist_y = np.sqrt((y[i]/y1-y[j]/y1)**2 / ((x[i]/x1-x[j]/x1)**2 + (y[i]/y1-y[j]/y1)**2)) * move_dist
                    #move_dist_x = move_dist/1.1
                    #move_dist_y = move_dist/1.1
                    
                    if x[i] < x[j]:
                        x[i] -= move_dist_x * x1
                        x[j] += move_dist_x * x1
                    else:
                        x[i] += move_dist_x * x1
                        x[j] -= move_dist_x * x1
                    
                    if y[i] < y[j]:
                        y[i] -= move_dist_y * y1
                        y[j] += move_dist_y * y1
                    else:
                        y[i] += move_dist_y * y1
                        y[j] -= move_dist_y * y1

    for i in index_list:
        for j in index_list:
            if i < j:
                temp = np.sqrt((x[i]/x1-x[j]/x1)**2 + (y[i]/y1-y[j]/y1)**2) - R[i] - R[j]
                if temp < 0:
                    iterate_sign = True




#plt.scatter(x, y, s=s)
#plt.show()

###########################################################################################
x_list = [xi for i,xi in zip(umapdf_DBSCAN2.index, x)]
y_list = [yi for i,yi in zip(umapdf_DBSCAN2.index, y)]
umapdf_DBSCAN2['adjusted_x'] = x_list
umapdf_DBSCAN2['adjusted_y'] = y_list
umapdf_DBSCAN2.to_csv(data_path_save + 'task_space_draw/' + 'umapdf_DBSCSN2_level1_user_count.csv')

save_obj(umapdf_DBSCAN2, 'umapdf_DBSCAN2_level1_user_count', data_path_save + 'task_space_draw/')
###########################################################################################
cluster_conclusion_adjusted = {0: 'Basic Web Dev',
1: 'Web Design' ,
2: 'Web Frameworks',
3: 'Desktop Apps',
4: 'Basic Programming\nConcepts',
5: 'Enterprise IT',
6: 'Enterprise and Web App Dev',
7: 'iOS',
8: 'Android',
9: 'SQL and Databases',
10: 'DevOps',
11: 'Cloud Computing',
12: 'Data Collection\nand Processing',
13: 'Networking',
14: 'Advanced Programming Concepts',
15: 'AI/ML',
16: "Statistics and\nData Analysis"
}

cluster_text_x = {0:3.85,
1:5.3,
2:4.9,
3:1.6,
4:1.0,
5:2.5,
6:3.8,
7:2.3,
8:2.5,
9:2,
10:3.4,
11:1.45,
12:1.02,
13:-0.6,
14:-0.4,
15:0.71,
16:-0.8
}
cluster_text_y = {0:5.2,
1:5.7,
2:3.2,
3:3.53,
4:2.2,
5:4.1,
6:2.76,
7:5,
8:6.7,
9:2.9,
10:1.6,
11:1.8,
12:1.15,
13:3.4,
14:3.75,
15:1.66,
16:0.91
}

df_dict_c = pd.Series(umapdf_DBSCAN2.node_color.values,index=umapdf_DBSCAN2.cluster_inc_all).to_dict()

dfl1 = [c for c in cluster_conclusion_adjusted.keys()]
dfl2 = [cluster_conclusion_adjusted[c] for c in dfl1]
dflx = [cluster_text_x[c] for c in dfl1]
dfly = [cluster_text_y[c] for c in dfl1]
dflc = [df_dict_c[c] for c in dfl1]
df_cluster_adjusted = pd.DataFrame.from_dict({'cluster': dfl1, 'description': dfl2, 'x':dflx, 'y':dfly, 'color':dflc})
df_cluster_adjusted.to_csv(data_path_save + 'task_space_draw/' + 'cluster_description_adjusted.csv')

save_obj(df_cluster_adjusted, 'cluster_conclusion_adjusted', data_path_save + 'task_space_draw/')
###########################################################################################
###########################################################################################
##! HN job data
###########################################################################################
import importlib
import job_gpt_35 #import the module here, so that it can be reloaded.
importlib.reload(job_gpt_35)
from job_gpt_35 import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

i = 0
job_ads = {}
with jsonlines.open(f'{data_path}hnhiring_jobs.json', 'r') as fcc_file:
    for line in fcc_file:
        for jl in line:
            job_ads[i] = jl
            i += 1
        
print(len(job_ads))

save_obj(job_ads,'job_ad_dict_from_HN', data_path_save + f'jobs/{data_label}/')
###########################################################################################
# from openai import OpenAI
# from tqdm import tqdm

# job_details_from_gpt35 = {}

# #client = OpenAI(api_key='')

# period_length = 100

# job_len = int(len(line)/period_length) + 1
# for i in range(453, job_len):
#     print(i)
#     gpt_dict = {}
#     job_list = line[i*period_length: (i + 1) * period_length]
#     for j, job in tqdm(enumerate(job_list)):
        
#         text = job['title'] + ' ' + job['details']
    
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo-0125",
#             response_format={ "type": "json_object" },
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant and expert in software industry designed to output JSON."},
#                 {"role": "user", "content": f"Here is a job ad:{text}. Could you extract in a json format the following information with required json key names:\n1. company name, with key name 'company_name'.\n2. bool of whether the ad is for more than one job, with key name 'multi_jobs'. 3. With key name 'jobs', build a list containing the information of each jobs: for each job, build a key-values pair with the job name as key name and value containing the following items: bool of whether this ad contain information of yearly salary values, with key name 'salary_index'; location, with key name 'location'; salary, with key name 'salary' and containing the following items: if the salary is a range, please give an average value with the key name 'salary_average', the minimum value with the key name 'salary_min', the maximum value with the key name 'salary_max'; if the salary is not a range, please give the value with the key name 'salary_average'; currency of salary, with key name 'currency'; bool of whether salary includes equity, with key name 'include_equity'; a list of skills required by the job, with key name 'skills'.\n Please only use the content in the advertisement and do not use other outside information."}
#                 ]
#         )
#         gpt_dict[str(j + i * period_length)] = response.choices[0].message.content

#     save_obj(gpt_dict, f'gpt_dict_{i}', data_path_save + 'gpt_advertisements/')

import importlib
import job_gpt_35 #import the module here, so that it can be reloaded.
importlib.reload(job_gpt_35)
from job_gpt_35 import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

data_label = 'HN_data_gpt35'

i = 0
job_ads = {}
with jsonlines.open(f'{data_path}hnhiring_jobs.json', 'r') as fcc_file:
    for line in fcc_file:
        i += 1


period_length = 100
job_len = int(len(line)/period_length) + 1    

gpt_dict = {}
for i in range(job_len):
    gpt_dict_temp = load_obj(f'gpt_dict_{i}', data_path_save + 'gpt_advertisements/')
    gpt_dict.update(gpt_dict_temp)



save_obj(gpt_dict, 'gpt_dict_all', data_path_save + 'gpt_advertisements/')
save_obj(gpt_dict, 'gpt_dict_all', data_path_save + f'jobs/{data_label}/')

###########################################################################################
import importlib
import job_gpt_35 #import the module here, so that it can be reloaded.
importlib.reload(job_gpt_35)
from job_gpt_35 import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

i = 0
with jsonlines.open(f'{data_path}hnhiring_jobs.json', 'r') as fcc_file:
    for job_sample in fcc_file:
        i += 1

gpt_dict = load_obj('gpt_dict_all', data_path_save + f'jobs/{data_label}/')
gpt_problem_list = [4900, 4960, 8349, 9783,14684,  16937, 19290, 21157, 23362, 31413, 35526, 37232, 41943, 42810, 43331, 43355, 45493, 45973, 49222]

salary_dataset = {}
skill_dataset = {}
job_salary_skill_dataset = {}
for ji, js in tqdm(enumerate(job_sample)):
    if ji not in gpt_problem_list:
        salary_dataset[str(ji)] = detect_salary_str(json.loads(gpt_dict[str(ji)]), js)
        skill_dataset[str(ji)] = detect_skill_str(json.loads(gpt_dict[str(ji)]))
        if salary_dataset[str(ji)]['job_names'] != skill_dataset[str(ji)]['job_names']:
            print(ji)

        for pid, p in enumerate(salary_dataset[str(ji)]['job_names']):
            job_name = str(ji) + '_' + str(pid) + '_' + p
            job_salary_skill_dataset[job_name] = {}
            job_salary_skill_dataset[job_name]['salary_bool'] = salary_dataset[str(ji)]['salary_bools'][pid]
            job_salary_skill_dataset[job_name]['salary_value'] = salary_dataset[str(ji)]['salary_values'][pid]
            job_salary_skill_dataset[job_name]['salary_currency'] = salary_dataset[str(ji)]['salary_currency'][pid]
            if len(skill_dataset[str(ji)]['skill_lists'][pid]) > 0:
                job_salary_skill_dataset[job_name]['skill_bool'] = True
            else:
                job_salary_skill_dataset[job_name]['skill_bool'] = False

            job_salary_skill_dataset[job_name]['skill_list'] = skill_dataset[str(ji)]['skill_lists'][pid]
            job_salary_skill_dataset[job_name]['year'] = js['date'][:4]
            

temp = job_salary_skill_dataset['25156_0_Database Developer']['skill_list'][0]
job_salary_skill_dataset['25156_0_Database Developer']['skill_list'] = [j for j in temp.keys()]

temp = job_salary_skill_dataset['42247_0_Full time, INTERNS/co-ops']['skill_list']
job_salary_skill_dataset['42247_0_Full time, INTERNS/co-ops']['skill_list'] = [j for j in temp[:-1]]
job_salary_skill_dataset['42247_0_Full time, INTERNS/co-ops']['skill_list'].append(temp[-1][0])

temp = job_salary_skill_dataset['37390_0_DevOps engineer']['skill_list']
job_salary_skill_dataset['37390_0_DevOps engineer']['skill_list'] = [list(j.values())[0] for j in temp]

save_obj(job_salary_skill_dataset, 'job_salary_skill_dataset', data_path_save + f'jobs/{data_label}/')
###########################################################################################
job_skill_dataset = {}
for j, jd in job_salary_skill_dataset.items():
    if jd['skill_bool'] and len(jd['skill_list']) >= 1:
        job_skill_dataset[j] = jd['skill_list']

save_obj(job_skill_dataset, f'job_skill_dataset', data_path_save + f'jobs/{data_label}/')

###########################################################################################
df_currency = ['usd', '$', 'gbp', '£', 'eur', '€', 'chf', 'cad', 'c$', 'euro', 'euros']
df_currency_short = ['USD', 'USD', 'GBP', 'GBP', 'EUR', 'EUR', 'CHF', 'CAD', 'CAD', 'EUR', 'EUR']
currency_dict = dict(zip(df_currency, df_currency_short))
for currency in [ 'GBP', 'CHF', 'CAD', 'EUR']:
    crcsv = pd.read_csv( data_path_save + 'jobs/' + f'USD_{currency}_history.csv')
    yr_currency = defaultdict(float)
    yr_count = defaultdict(int)
    for i in range(len(crcsv)):
        yr = crcsv.iloc[i]['日期'][:4]
        yr_currency[yr] += float(crcsv.iloc[i]['收盘'])
        yr_count[yr] += 1

    for yr in yr_count.keys():
        yr_currency[yr] = yr_currency[yr]/yr_count[yr]

    save_obj(yr_currency, f'{currency}_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')

save_obj({str(yr):1 for yr in range(2018,2025)}, f'USD_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')


df_currency = ['usd', '$', 'gbp', '£', 'eur', '€', 'chf', 'cad', 'c$', 'euro', 'euros']
df_currency_short = ['USD', 'USD', 'GBP', 'GBP', 'EUR', 'EUR', 'CHF', 'CAD', 'CAD', 'EUR', 'EUR']
currency_dict = dict(zip(df_currency, df_currency_short))

gbp_yr_currency = load_obj(f'GBP_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')
eur_yr_currency = load_obj(f'EUR_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')
chf_yr_currency = load_obj(f'CHF_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')
cad_yr_currency = load_obj(f'CAD_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')
usd_yr_currency = load_obj(f'USD_USD_rate_by_year', data_path_save + f'jobs/{data_label}/')
currency_rate_all = {'USD':usd_yr_currency, 'GBP':gbp_yr_currency, 'CHF':chf_yr_currency, 'CAD':cad_yr_currency, 'EUR':eur_yr_currency}

currency_dict = dict(zip(df_currency, df_currency_short))

job_salary_regression_dict = {}
job_skill_regression_dict = {}
job_year_regression_dict = {}
for j, jd in job_salary_skill_dataset.items():
    if jd['skill_bool'] and jd['salary_bool'] and len(jd['skill_list']) >= 1:
        if jd['salary_currency'] in currency_dict:
            cr = currency_rate_all[currency_dict[jd['salary_currency']]][jd['year']]
            s = jd['salary_value'] / cr
            if s > 30000 and s < 1000000:
                job_salary_regression_dict[j] = s
                job_year_regression_dict[j] = jd['year']
                job_skill_regression_dict[j] = jd['skill_list']
        
save_obj(job_salary_regression_dict, f'regression_job_salary_dict', data_path_save + f'jobs/{data_label}/')
save_obj(job_skill_regression_dict, f'regression_job_skill_dict', data_path_save + f'jobs/{data_label}/')
save_obj(job_year_regression_dict, f'regression_job_year_dict', data_path_save + f'jobs/{data_label}/')
###########################################################################################
###########################################################################################
##! job data analysis
###########################################################################################
import importlib
import job_data_nlp_processing_gptdesp
importlib.reload(job_data_nlp_processing_gptdesp)
from job_data_nlp_processing_gptdesp import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {t[0]:t[1] for t in tag_count}

device = 1
job_skill_dataset = load_obj(f'job_skill_dataset', data_path_save + f'jobs/{data_label}/')
bert_skill_embedding_gptdesp(job_skill_dataset, data_path_save, device, data_label)

for level in range(1,2):
    print('level:  ', level)
    tag_count_bool = False

    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    
    #{'task_id':list_c, 'tags':list_ts, 'task_description_30_words': list_30, 'task_description_20_words': list_20, 'task_description_10_words':list_10, 'task_description_5_words':list_5, 'task_description_conclude': list_conclude}
    task_description = pd.read_csv(data_path_save + 'networks/probability/' + f'task_description_30_20_10_5_community_core_with_cut_level_{level}.csv')
    
    dict_task_description = pd.Series(task_description.task_description_30_words.values,index=task_description.task_id).to_dict()

    bert_task_description_embedding_gptdesp(community_unweighted_level, data_path_save, device, level, data_label, dict_task_description)
    
    occupation_embedding_dict = load_obj(f'gptdesp_job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    occupation_task_distance_matrix_gptdesp(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, data_label)
###########################################################################################
import importlib
import job_data_nlp_processing_gptdesp
importlib.reload(job_data_nlp_processing_gptdesp)
from job_data_nlp_processing_gptdesp import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

density_user_label = 'all_threshold_user'

skill_similarity_threshold = 0.3

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

    occupation_task_similarity_matrix = load_obj(f'gptdesp_job_task_similarity_matrix_level_{level}', data_path_save + f'jobs/{data_label}/')

    ##! build training and target
    print('build training and target')
    community_list_std_no_empty = [c for c in community_list_std if len(community_unweighted_level[c]) > 0]
    job_task_training_vector, job_task_training_set, job_task_target_set = get_job_community_set_gptdesp(occupation_task_similarity_matrix ,community_list_std, community_list_std_no_empty, skill_similarity_threshold)

    #job_task_prediction = predict_user_community(cc_pmi_matrix, job_task_training_vector)
    job_task_prediction = predict_user_community_ignore_target_gptdesp(cc_pmi_matrix, job_task_training_vector, community_list_std, job_task_target_set)
    community_prediction, cp_all = examine_prediction_gptdesp(job_task_prediction, job_task_target_set, job_task_training_set, community_list_std)

    save_obj(cp_all, f'gptdesp_cp_all_{level}', data_path_save + f'jobs/{data_label}/')
###########################################################################################
import importlib
import wage_regression_gpt_gptdesp #import the module here, so that it can be reloaded.
importlib.reload(wage_regression_gpt_gptdesp)
from wage_regression_gpt_gptdesp import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

data_label = 'HN_data_gpt35'

tag_count = load_obj('tag_count_all', data_path)
tag_count_dict = {t[0]:t[1] for t in tag_count}

device = 1
job_skill_dataset = load_obj(f'regression_job_skill_dict', data_path_save + f'jobs/{data_label}/')
bert_skill_embedding_regression_gptdesp(job_skill_dataset, data_path_save, device, data_label)

for level in range(1,2):
    print('level:  ', level)
    tag_count_bool = False

    community_unweighted_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    
    #{'task_id':list_c, 'tags':list_ts, 'task_description_30_words': list_30, 'task_description_20_words': list_20, 'task_description_10_words':list_10, 'task_description_5_words':list_5, 'task_description_conclude': list_conclude}
    task_description = pd.read_csv(data_path_save + 'networks/probability/' + f'task_description_30_20_10_5_community_core_with_cut_level_{level}.csv')
    
    dict_task_description = pd.Series(task_description.task_description_30_words.values,index=task_description.task_id).to_dict()

    bert_task_description_embedding_regression_gptdesp(community_unweighted_level, data_path_save, device, level, data_label, dict_task_description)
    
    occupation_embedding_dict = load_obj(f'gptdesp_regression_job_skill_embedding', data_path_save + f'jobs/{data_label}/')
    occupation_task_distance_matrix_regression_gptdesp(occupation_embedding_dict, community_unweighted_level, community_list_std, data_path_save, level, data_label)
###########################################################################################
import importlib
import wage_regression_gpt_gptdesp #import the module here, so that it can be reloaded.
importlib.reload(wage_regression_gpt_gptdesp)
from wage_regression_gpt_gptdesp import *


data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

skill_similarity_threshold = 0.3
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

    df = build_regression_dataframe_gptdesp(data_path_save, level, community_list_std, community_unweighted_level, skill_similarity_threshold,skill_length_threshold,so_yearlist, task_salary_name, data_label, density_user_label, salary_type)
    df.to_csv(data_path_save + f'jobs/{data_label}/gptdesp_{period_label}_{so_yearlist_salary}_{sv_yearlist_salary}_df_regression_level_{level}_topn_{topn}_{sample_user_label}.csv')
    
print(f'gptdesp_{period_label}_{so_yearlist_salary}_{sv_yearlist_salary}_df_regression_level_{level}_topn_{topn}_{sample_user_label}.csv')
###########################################################################################
###########################################################################################
##! SO user task prediction
###########################################################################################
import importlib
import task_prediction #import the module here, so that it can be reloaded.
importlib.reload(task_prediction)
from task_prediction import *

data_path = 'data_files/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'


density_user_label = 'half_user'

##! reverse the user bool
half_user_bool = load_obj('half_user_bool', data_path_save + f'vote_regression_together/')
all_threshold_user_bool = load_obj('all_threshold_user_bool', data_path_save + f'vote_regression_together/')

##! 用来预测的user部分
user_bool_reverse = defaultdict(bool)
for u in all_threshold_user_bool.keys():
    if not half_user_bool[u]:
        user_bool_reverse[u] = True 

year_period = 2

##! label
for level in [1,2,3]:
    ##! load network and community
    tag_bool_core = load_obj(f"core_bool_with_cut_level{level}", data_path_save + 'networks/probability/')
    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    community_core_level = load_obj(f"community_core_with_cut_level{level}", data_path_save + 'networks/probability/')
    tag_community_dict = {t:ic for ic,c in community_core_level.items() for t in c}

    cc_pmi_matrix = load_obj(f'cc_pmi_matrix_normalized_{density_user_label}_2008_2023_level_{level}', data_path_save + f'vote_regression_together/user_task_collection/')

    for yr in range(2008 + year_period + 1, 2024):

        yr_list = [yr - year_period + i for i in range(year_period)]
        print(yr_list)
        
        ##! build user vector
        print(f'================================= build source {yr_list} =================================')
        year_bool = {str(yrtemp):False for yrtemp in range(2008, 2024)}
        for yrtemp in yr_list:
            year_bool[str(yrtemp)] = True

        user_community_set_source, user_community_vector_binary_source = get_user_community_set_from_sample(year_bool, user_bool_reverse, data_path_save, community_list_core_std, level)

        ##! make prediction
        print(f"================================= make prediction {yr_list} =================================")
        year_bool = {str(yrtemp):False for yrtemp in range(2008, 2024)}
        year_bool[str(yr)] = True

        user_community_set_target, _ = get_user_community_set_from_sample(year_bool, user_bool_reverse, data_path_save, community_list_core_std, level)

        user_community_prediction = predict_user_community(cc_pmi_matrix, user_community_vector_binary_source)
        community_prediction, cp_all = examine_prediction(user_community_prediction, user_community_set_target, user_community_set_source, community_list_core_std)

        print(len(cp_all))

        save_obj(cp_all, f'cp_all_year_{yr}_level_{level}', data_path_save + 'density_prediction/')

###########################################################################################
###########################################################################################
##! Rscript to run
###########################################################################################
##! Run Rscript
##! Rscript vote_regression_comparison.R
##! Rscript value_regression_entry_all_language_2025.R
##! Rscript value_regression_entry_python_2025.R


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
