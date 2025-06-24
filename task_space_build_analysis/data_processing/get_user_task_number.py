import importlib
import user_task_collection #import the module here, so that it can be reloaded.
importlib.reload(user_task_collection)
from user_task_collection import *
random.seed(1730)

data_path = '/home/xiangnan/task_space_code/task_space_data/'
#data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

level = 1

community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')

sample_percent_label = 10

save_name = data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_level_{level}_{sample_percent_label}_percent_all_language.csv'

df_all = pd.read_csv(save_name)

print(len(df_all.user_id.unique()))
print(len(df_all.task.unique()))

del df_all



save_name = data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + f'df_user_task_entry_exit_salary_task_with_density_level_{level}_{sample_percent_label}_percent_python.csv'

df_python = pd.read_csv(save_name)

print(len(df_python.user_id.unique()))
print(len(df_python.task.unique()))

del df_python