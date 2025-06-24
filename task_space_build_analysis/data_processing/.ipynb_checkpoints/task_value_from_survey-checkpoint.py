import json
import random
from tqdm import tqdm
from pickle_file import load_obj, save_obj
import pandas as pd
import numpy as np
import jsonlines
from scipy.sparse import csr_matrix, hstack, vstack
from collections import Counter, defaultdict


def get_country_count_from_sv(data_path, country_dict):

    df = pd.read_csv(data_path + "combined_survey_data.csv")

    sv_country_count = {str(y):defaultdict(int) for y in list(df['year'].value_counts().keys())}

    for i, y, acr, c, country in zip(list(df.index.values), df['year'].values.tolist(), df['annual_comp_rounded'].values.tolist(), df['currency'].values.tolist(), df['country'].values.tolist()):
        if not pd.isna(y) and not pd.isna(acr) and not pd.isna(c) and c != 'Bitcoin (btc)' and country in country_dict:

            sv_country_count[str(y)][country_dict[country]] += 1

    return sv_country_count



def get_sv_salary(data_path, data_path_save, country_count, country_dict, cc_threshold, period_label):

    df = pd.read_csv(data_path + "combined_survey_data.csv")

    df_currency = ['Euros (€)', 'British pounds sterling (£)', 'Canadian dollars (C$)', 'Polish zloty (zl)', 'Australian dollars (A$)', 'Russian rubles (?)', 'Brazilian reais (R$)', 'Swedish kroner (SEK)', 'Swiss francs', 'South African rands (R)', 'Mexican pesos (MXN$)', 'Chinese yuan renminbi (¥)', 'Singapore dollars (S$)']

    df_currency_short = ['EUR', 'GBP', 'CAD', 'PLN', 'AUD', 'RUB', 'BRL', 'SEK', 'CHF', 'ZAR', 'MXN', 'CNY', 'SGD']

    df_currency_convert = dict(zip(df_currency, df_currency_short))


    currency_rate = {}
    currency_rate['USD'] = {str(y):1 for y in list(df['year'].value_counts().keys())}
    currency_rate['U.S. dollars ($)'] = {str(y):1 for y in list(df['year'].value_counts().keys())}

    def get_currency_rate(dfcc):
        crdict = {str(y):0 for y in list(df['year'].value_counts().keys())}
        crcsv = pd.read_csv(data_path + 'currency_rate/' + f'{dfcc}_USD历史数据.csv')
        cr_temp = {str(y):[] for y in list(df['year'].value_counts().keys())}
        for t, c in zip(crcsv['日期'].values.tolist(), crcsv['收盘'].values.tolist()):
            y = t[0:4]
            if y > '2010':
                cr_temp[y].append(c)

        for y,c in cr_temp.items():
            crdict[y] = sum(c)/len(c)

        return crdict


    for c in df_currency:
        print(c)
        currency_rate[c] = get_currency_rate(df_currency_convert[c])


    df_currency_2 = ['Indian rupees (?)', 'Japanese yen (¥)']

    df_currency_short_2 = ['INR', 'JPY']

    df_currency_convert.update(dict(zip(df_currency_2, df_currency_short_2)))

    def get_currency_rate_2(dfcc):

        crdict = {str(y):0 for y in list(df['year'].value_counts().keys())}
        crcsv = pd.read_csv(data_path + 'currency_rate/' + f'USD_{dfcc}历史数据.csv')
        cr_temp = {str(y):[] for y in list(df['year'].value_counts().keys())}
        for t, c in zip(crcsv['日期'].values.tolist(), crcsv['收盘'].values.tolist()):
            y = t[0:4]
            if y > '2010':
                cr_temp[y].append( 1 / c )

        for y,c in cr_temp.items():
            crdict[y] = sum(c)/len(c)

        return crdict


    for c in df_currency_2:
        print(c)
        currency_rate[c] = get_currency_rate_2(df_currency_convert[c])


    salary_list= []
    sv_id_list = []
    year_list = []
    c_list = []
    currency_list = []

    for i, y, acr, c, country in zip(list(df.index.values), df['year'].values.tolist(), df['annual_comp_rounded'].values.tolist(), df['currency'].values.tolist(), df['country'].values.tolist()):
        if not pd.isna(y) and not pd.isna(acr) and not pd.isna(c) and c != 'Bitcoin (btc)' and country in country_dict and country_count[str(y)][country_dict[country]] > cc_threshold:
            sv_id_list.append(i)
            year_list.append(str(y))
            salary_list.append(acr * currency_rate[c][str(y)])
            c_list.append(country_dict[country])
            currency_list.append(c)
            
    df_salary = pd.DataFrame.from_dict({'sv_id':sv_id_list, 'year':year_list, 'salary':salary_list, 'country':c_list, 'currency':currency_list})

    df_salary.to_csv(data_path_save + f'surveys/country/{period_label}/sv_salary_df.csv')

    resid_dict_by_year = {yr:{} for yr in df_salary['year'].values.tolist()}
    sv_country_dict = {yr:{} for yr in df_salary['year'].values.tolist()}
    
    for i, svid, yr, country, salary in tqdm(zip(list(df_salary.index.values), df_salary['sv_id'].values.tolist(), df_salary['year'].values.tolist(), df_salary['country'].values.tolist(), df_salary['salary'].values.tolist())):
        resid_dict_by_year[yr][svid] = salary
        sv_country_dict[yr][svid] = country_dict[country]

    save_obj(resid_dict_by_year, 'survey_user_salary_dict_by_year_country', data_path_save + f'surveys/country/{period_label}/')
    save_obj(sv_country_dict, 'survey_user_country_dict_by_year', data_path_save + f'surveys/country/{period_label}/')
    
    return



def get_sv_salary_log(data_path, data_path_save, country_count, country_dict, cc_threshold, period_label):

    df = pd.read_csv(data_path + "combined_survey_data.csv")

    df_currency = ['Euros (€)', 'British pounds sterling (£)', 'Canadian dollars (C$)', 'Polish zloty (zl)', 'Australian dollars (A$)', 'Russian rubles (?)', 'Brazilian reais (R$)', 'Swedish kroner (SEK)', 'Swiss francs', 'South African rands (R)', 'Mexican pesos (MXN$)', 'Chinese yuan renminbi (¥)', 'Singapore dollars (S$)']

    df_currency_short = ['EUR', 'GBP', 'CAD', 'PLN', 'AUD', 'RUB', 'BRL', 'SEK', 'CHF', 'ZAR', 'MXN', 'CNY', 'SGD']

    df_currency_convert = dict(zip(df_currency, df_currency_short))


    currency_rate = {}
    currency_rate['USD'] = {str(y):1 for y in list(df['year'].value_counts().keys())}
    currency_rate['U.S. dollars ($)'] = {str(y):1 for y in list(df['year'].value_counts().keys())}

    def get_currency_rate(dfcc):
        crdict = {str(y):0 for y in list(df['year'].value_counts().keys())}
        crcsv = pd.read_csv(data_path + 'currency_rate/' + f'{dfcc}_USD历史数据.csv')
        cr_temp = {str(y):[] for y in list(df['year'].value_counts().keys())}
        for t, c in zip(crcsv['日期'].values.tolist(), crcsv['收盘'].values.tolist()):
            y = t[0:4]
            if y > '2010':
                cr_temp[y].append(c)

        for y,c in cr_temp.items():
            crdict[y] = sum(c)/len(c)

        return crdict


    for c in df_currency:
        print(c)
        currency_rate[c] = get_currency_rate(df_currency_convert[c])


    df_currency_2 = ['Indian rupees (?)', 'Japanese yen (¥)']

    df_currency_short_2 = ['INR', 'JPY']

    df_currency_convert.update(dict(zip(df_currency_2, df_currency_short_2)))

    def get_currency_rate_2(dfcc):

        crdict = {str(y):0 for y in list(df['year'].value_counts().keys())}
        crcsv = pd.read_csv(data_path + 'currency_rate/' + f'USD_{dfcc}历史数据.csv')
        cr_temp = {str(y):[] for y in list(df['year'].value_counts().keys())}
        for t, c in zip(crcsv['日期'].values.tolist(), crcsv['收盘'].values.tolist()):
            y = t[0:4]
            if y > '2010':
                cr_temp[y].append( 1 / c )

        for y,c in cr_temp.items():
            crdict[y] = sum(c)/len(c)

        return crdict


    for c in df_currency_2:
        print(c)
        currency_rate[c] = get_currency_rate_2(df_currency_convert[c])


    salary_list= []
    sv_id_list = []
    year_list = []
    c_list = []
    currency_list = []

    for i, y, acr, c, country in zip(list(df.index.values), df['year'].values.tolist(), df['annual_comp_rounded'].values.tolist(), df['currency'].values.tolist(), df['country'].values.tolist()):
        if not pd.isna(y) and not pd.isna(acr) and not pd.isna(c) and c != 'Bitcoin (btc)' and country in country_dict and country_count[str(y)][country_dict[country]] > cc_threshold:
            sv_id_list.append(i)
            year_list.append(str(y))
            salary_list.append(np.log(acr * currency_rate[c][str(y)]))
            c_list.append(country_dict[country])
            currency_list.append(c)
            
    df_salary = pd.DataFrame.from_dict({'sv_id':sv_id_list, 'year':year_list, 'salary':salary_list, 'country':c_list, 'currency':currency_list})

    df_salary.to_csv(data_path_save + f'surveys/country/{period_label}/sv_salary_df.csv')

    resid_dict_by_year = {yr:{} for yr in df_salary['year'].values.tolist()}
    sv_country_dict = {yr:{} for yr in df_salary['year'].values.tolist()}
    
    for i, svid, yr, country, salary in tqdm(zip(list(df_salary.index.values), df_salary['sv_id'].values.tolist(), df_salary['year'].values.tolist(), df_salary['country'].values.tolist(), df_salary['salary'].values.tolist())):
        resid_dict_by_year[yr][svid] = salary
        sv_country_dict[yr][svid] = country_dict[country]

    save_obj(resid_dict_by_year, 'survey_user_salary_dict_by_year_country', data_path_save + f'surveys/country/{period_label}/')
    save_obj(sv_country_dict, 'survey_user_country_dict_by_year', data_path_save + f'surveys/country/{period_label}/')
    
    return



def get_sv_tags(data_path, data_path_save, period_label):

    tag_rename_dict = load_obj('tag_rename_dict',data_path)
    tagset_all = load_obj('tagset_all', data_path)

    ##! tag_projection
    ##! tag_projection
    tag_projection = {}
    df_str = pd.read_csv(data_path + 'survey_tag_reflection.csv')
    for st, tp1, tp2 in zip(df_str['survey_tag'].values.tolist(), df_str['tag_projection1'].values.tolist(), df_str['tag_projection2'].values.tolist()):
        if not pd.isna(tp2):
            tag_projection[st] = [tp1, tp2]

        else:
            tag_projection[st] = [tp1]


    ##! get the tags
    ##! get the tags
    df = pd.read_csv(data_path + "combined_survey_data.csv")
    user_survey_tag_all_by_year = {str(y):{} for y in list(df['year'].value_counts().keys())}
    user_survey_tag_length_all_by_year = {str(y):{} for y in list(df['year'].value_counts().keys())}

    survey_tag_list_by_year = {str(y):[] for y in list(df['year'].value_counts().keys())}

    for idx, ts, yr in tqdm(zip(list(df.index.values), df['list_of_langs'].values.tolist(), df['year'].values.tolist())):
        if len(ts) > 2 and not pd.isna(yr):

            temp = [t.lstrip(" ").rstrip(" ").lstrip("'").rstrip("'").lower().replace('-','_') for t in ts[1:-1].replace('"',"'").split("', '")]
            temp1 = [t.lstrip(' ').rstrip(' ') for tt in temp for t in tt.split('/')]
            temp2 = [t.lstrip(' ').rstrip(' ') for tt in temp1 for t in tt.split(',')]
            
            tempt = [t.lower().replace('-','_').replace(' ','_') for t in temp2]
            temptt = [t for t in tempt if tag_rename_dict.get(t) is None] + [tag_rename_dict[t] for t in tempt if tag_rename_dict.get(t) is not None]
            
            utemp = []
            for t in temptt:
                if tag_projection.get(t) is not None:
                    utemp += tag_projection[t]
                else:
                    utemp += [t]

            user_survey_tag_all_by_year[str(yr)][idx] = set([t for t in utemp if len(t) > 0 and t != 'this_is_an_empty_tag'])
            user_survey_tag_length_all_by_year[str(yr)][idx] = len(set([t for t in utemp if len(t) > 0 and t != 'this_is_an_empty_tag']))

            survey_tag_list_by_year[str(yr)] += list(set([t for t in utemp if len(t) > 0 and t != 'this_is_an_empty_tag']))

    from collections import Counter
    survey_tag_count = {k:sorted(dict(Counter(survey_tag_list)).items(), key = lambda kv:(-kv[1], kv[0])) for k, survey_tag_list in survey_tag_list_by_year.items()}

    survey_tag_threshold_by_year = {}
    for y,ust in survey_tag_count.items():
        survey_tag_threshold_by_year[y] = set([s[0] for s in ust if s[1] > 5])
        print(y, 'number of tags: ', len(survey_tag_threshold_by_year[y]), 'number of tags overlaped with so: ', len(survey_tag_threshold_by_year[y].intersection(tagset_all)))

    save_obj(user_survey_tag_all_by_year, 'user_survey_tag_all_by_year', data_path_save + f'surveys/country/{period_label}/')
    save_obj(user_survey_tag_length_all_by_year, 'user_survey_tag_length_all_by_year', data_path_save + f'surveys/country/{period_label}/')
    save_obj(survey_tag_threshold_by_year, 'survey_tag_threshold_by_year', data_path_save + f'surveys/country/{period_label}/')


def get_sv_salary_tags(data_path, data_path_save, period_label):

    ##! load data
    ##! load data
    salary_dict_by_year = load_obj('survey_user_salary_dict_by_year_country', data_path_save + f'surveys/country/{period_label}/')
    user_survey_tag_all_by_year = load_obj('user_survey_tag_all_by_year', data_path_save + f'surveys/country/{period_label}/')
    tagset_all = load_obj('tagset_all', data_path)
    survey_tag_threshold_by_year = load_obj('survey_tag_threshold_by_year', data_path_save + f'surveys/country/{period_label}/')
    sv_country_dict = load_obj('survey_user_country_dict_by_year', data_path_save + f'surveys/country/{period_label}/')

    ##! people with salary and year_experience
    ##! people with salary and year_experience
    salary_user_key = set([i for isa in list(salary_dict_by_year.values()) for i in list(isa.keys())])

    ##! collect the data
    ##! collect the data
    salary_user_survey_tag_by_year = {str(k):{} for k in range(2011,2024)}
    salary_user_survey_tag_length_by_year = {str(k):{} for k in range(2011,2024)}
    salary_user_survey_tag_country_by_year = {str(k):{} for k in range(2011,2024)}

    for yr, utls in tqdm(user_survey_tag_all_by_year.items()):
        index_temp = set(utls.keys()).intersection(salary_user_key)
        for u in tqdm(index_temp):
            temp = utls[u].intersection(tagset_all)
            temp1 = temp.intersection(set(survey_tag_threshold_by_year[yr]))
            if len(temp1) > 0:
                salary_user_survey_tag_by_year[yr][u] = set([t for t in temp1])
                salary_user_survey_tag_country_by_year[yr][u] = sv_country_dict[yr][u]
                salary_user_survey_tag_length_by_year[yr][u] = len(temp1)

    print("=================================================")

    for yr, st in salary_user_survey_tag_by_year.items():
        print(yr, len(st), len(user_survey_tag_all_by_year[yr]))

    sv_tagset_dict = {str(k):set([]) for k in range(2011,2024)}
    for yr, utls in salary_user_survey_tag_by_year.items():
        for tls in utls.values():
            sv_tagset_dict[yr] = sv_tagset_dict[yr] | tls

    sv_taglist_std_dict = {k:list(v) for k,v in sv_tagset_dict.items()}
    print([(k,len(v)) for k,v in sv_taglist_std_dict.items()])

    save_obj(salary_user_survey_tag_by_year, 'salary_user_survey_tag_threshold_by_year_with_country', data_path_save + f'surveys/country/{period_label}/')
    save_obj(salary_user_survey_tag_length_by_year, 'salary_user_survey_tag_length_threshold_by_year_with_country', data_path_save + f'surveys/country/{period_label}/')
    save_obj(salary_user_survey_tag_country_by_year, 'salary_user_survey_tag_country_threshold_by_year', data_path_save + f'surveys/country/{period_label}/')
    save_obj(sv_taglist_std_dict, 'salary_user_survey_taglist_std_by_year_with_country', data_path_save + f'surveys/country/{period_label}/')

    return

    

def product_so_tags_sv_topn(so_yearlist, sv_yearlist, period_label, data_path_save, topn, sample_user_label):

    import time

    ##! survey tag list std
    print('survey tag list std')
    sv_taglist_std_dict = load_obj('salary_user_survey_taglist_std_by_year_with_country', data_path_save + f'surveys/country/{period_label}/')
    tag_std = list(set([t for yr in sv_yearlist for t in sv_taglist_std_dict[str(yr)]]))
    tag_std_index = {t:i for i,t in enumerate(tag_std)}
    tag_std_set = set(tag_std)
    print(len(tag_std_set))

    ##! build SV user tag
    print('build SV user tag')
    salary_user_survey_tag = load_obj('salary_user_survey_tag_threshold_by_year_with_country', data_path_save + f'surveys/country/{period_label}/')
    sv_user_tag_dict = {u:utst for yr in sv_yearlist for u,utst in salary_user_survey_tag[str(yr)].items()}
    sv_user_index = {u:i for i,u in enumerate(sv_user_tag_dict.keys())}
    sv_index_user = {i:u for u, i in sv_user_index.items()}

    ##! build SV salary matrix
    resid_dict_by_year = load_obj('survey_user_salary_dict_by_year_country', data_path_save + f'surveys/country/{period_label}/')
    sv_salary_dict = {u:s for yr in sv_yearlist for u,s in resid_dict_by_year[str(yr)].items()}

    ##! build SV user-tag sparse matrix, row: tags, columns: SV-users
    print('build SV user-tag sparse matrix')

    row_SV = []
    col_SV = []
    data_SV = []
    for u, tagset in tqdm(sv_user_tag_dict.items()):
        for t in tagset:
            row_SV.append(tag_std_index[t])
            col_SV.append(sv_user_index[u])
            data_SV.append(1/len(tagset))

    SV_Matrix = csr_matrix((data_SV, (row_SV, col_SV)), shape = (len(tag_std_index), len(sv_user_index)))


    ##! build SO user tag
    print('build SO user tag')
    so_sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    so_user_tag_dict = defaultdict(set)
    for yr in tqdm(so_yearlist):
        user_tag_set_temp = load_obj(f'tag_user_set_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, utst in user_tag_set_temp.items():
            if so_sample_user_bool[u]:
                utst_temp = utst.intersection(tag_std_set)
                if len(utst_temp) > 0:
                    so_user_tag_dict[u] = so_user_tag_dict[u] | utst_temp

    so_user_list = [u for u in so_user_tag_dict.keys()]
    so_user_index = {u:i for i,u in enumerate(so_user_list)}
    index_to_so_user = {i:u for u,i in so_user_index.items()}

    matrix_length = 50000
    SO_list_index = [(i*matrix_length, (i+1)*matrix_length) for i in range(int(len(so_user_list) / matrix_length))] + [(int(len(so_user_list) / matrix_length) * matrix_length,len(so_user_list))]

    
    SO_user_salary = {}
    ##! build SO user-tag sparse matrix, rows: SO-users, columns: tags,
    print('build SO user-tag sparse matrix')
    for save_i, index_pair in enumerate(SO_list_index):
        row_SO = []
        col_SO = []
        data_SO = []
        for u in so_user_list[index_pair[0]:index_pair[1]]:    
            for t in so_user_tag_dict[u]:
                col_SO.append(tag_std_index[t])
                row_SO.append(so_user_index[u] - save_i*matrix_length)
                data_SO.append(1/len(so_user_tag_dict[u]))

        SO_Matrix = csr_matrix((data_SO, (row_SO, col_SO)), shape = (len(so_user_list[index_pair[0]:index_pair[1]]), len(tag_std_index)))

        start = time.time()
        print(f'SO SV production {save_i}')
        SO_SV_product = SO_Matrix @ SV_Matrix
        print(SO_SV_product.shape)
        end = time.time()
        print('wall-clock running time : ',end - start)
        del SO_Matrix

        print('get SO user salarys')
        num_rows = SO_SV_product.shape[0]
        for i in tqdm(range(num_rows)):
            le = SO_SV_product.indptr[i]
            ri = SO_SV_product.indptr[i + 1]
            n_row_pick = min(topn, ri - le)
            
            indices_temp = le + np.argpartition(SO_SV_product.data[le:ri], -n_row_pick)[-n_row_pick:]
            topindex_salary = np.array([sv_salary_dict[sv_index_user[ui]] for ui in SO_SV_product.indices[indices_temp]]).T
            sumtemp = sum(SO_SV_product.data[indices_temp])
            topdata = np.array([tt/sumtemp for tt in SO_SV_product.data[indices_temp]])
            SO_user_salary[index_to_so_user[i + matrix_length * save_i]] = topdata@topindex_salary


        end = time.time()
        print('wall-clock running time : ',end - start)
        del SO_SV_product

    save_obj(SO_user_salary, f'{so_yearlist}_{sv_yearlist}_SO_salary_match_topn_{topn}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')
    save_obj(so_user_list, f'{so_yearlist}_{sv_yearlist}_so_user_list_match_topn_{topn}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')

    return



def calculate_task_salary_topn(data_path_save, period_label, so_year_list, sv_year_list, level, topn, sample_user_label):

    from sklearn.preprocessing import normalize

    SO_user_salary = load_obj(f'{so_year_list}_{sv_year_list}_SO_salary_match_topn_{topn}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')
    so_user_list = load_obj(f'{so_year_list}_{sv_year_list}_so_user_list_match_topn_{topn}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')


    community_list_core_std = load_obj(f"community_list_std_core_cut_level{level}", data_path_save + 'networks/probability/')
    so_sample_user_bool = load_obj(f'{sample_user_label}_bool', data_path_save + f'vote_regression_together/')
    
    ##! SO user task set
    task_user_bool = {t:defaultdict(bool) for t in community_list_core_std}
    user_task_count_all = {}
    user_task_totalcount = defaultdict(int)
    user_bool_temp = defaultdict(bool)
    for yr in so_year_list:
        user_task_count_temp = load_obj(f'user_task_count_by_single_year_level_{level}_{yr}', data_path_save + 'vote_regression_together/user_c_l_list/')
        for u, ts in user_task_count_temp.items():
            if so_sample_user_bool[u]:
                if not user_bool_temp[u]:
                    user_bool_temp[u] = True
                    user_task_count_all[u] = defaultdict(int)

                for t, tc in ts.items():
                    user_task_count_all[u][t] += tc
                    user_task_totalcount[u] += tc
                    task_user_bool[t][u] = True

    task_salary = {}
    for t in tqdm(community_list_core_std):
        task_user_vector_normalized = normalize(np.array([[user_task_count_all[u][t]/user_task_totalcount[u] if task_user_bool[t][u] else 0 for u in so_user_list]]), axis=1, norm='l1').T
        task_salary[t] = (np.array([SO_user_salary[u] for u in so_user_list])@task_user_vector_normalized)[0]

    save_obj(task_salary, f'{so_year_list}_{sv_year_list}_task_salary_topn_{topn}_level_{level}_{sample_user_label}', data_path_save + f'surveys/country/{period_label}/')

    return task_salary
