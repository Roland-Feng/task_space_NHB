##Read and Load files

#import pickle5 as pickle
import pickle
def save_obj(obj, name, data_path_save = 'obj/'):
    with open(data_path_save + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def load_obj(name, data_path_load = 'obj/'):
    with open(data_path_load + name + '.pkl', 'rb') as f:
        return pickle.load(f)
