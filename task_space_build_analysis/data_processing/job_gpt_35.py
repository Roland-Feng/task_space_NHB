import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
import json



def judge_number(number_str):
    s = ''
    for n in number_str:
        if n!=',':
            s += n

    if len(s) > 0:
        if s[-1] == 'k':
            s = s[:-1] + '000'
            
        if s.isdigit():
            if int(s) < 1000:
                return int(s) * 1000
            else:
                return int(s)
        else:
            return -100869573
        
    else:
        return -100869573
    
    
def judge_average_in_sample(n, j_sample):
    nk = str(int(n/1000)) + 'k'

    ncomma = str(n)[:-3] + ',' + str(n)[-3:]
    
    jsl = j_sample.lower()
    if str(n) in jsl or ncomma in jsl:
        return True
    
    elif nk in jsl:
        return True
    else:
        return False

def judge_maximal_in_sample(n, j_sample):
    nk = str(int(n/1000)) + 'k'
    ncomma = str(n)[:-3] + ',' + str(n)[-3:]
    
    jsl = j_sample.lower()
    if str(n) in jsl or ncomma in jsl:
        return True
    
    elif nk in jsl:
        return True
    else:
        return False
    

def judge_minimal_without_k(n, j_sample):
    nk = str(int(n/1000))
    ncomma = str(n)[:-3] + ',' + str(n)[-3:]
    
    jsl = j_sample.lower()
    if str(n) in jsl or ncomma in jsl:
        return True
    
    elif nk in jsl:
        return True
    else:
        return False
    



def detect_salary_str(json_str, j_sample):
    job_names = []
    salary_bools = []
    salary_values = []
    salary_currency = []
    job_city = []
    job_country = []
    salary_time = []
    full_time_index = []
    one_year_salary_index = []

    text = j_sample['title'] + j_sample['details']
    if 'jobs' in json_str:
        if type(json_str['jobs']) is dict and json_str['multi_jobs']:
            json_jobs = [{jt:tt} for jt, tt in json_str['jobs'].items()]

        elif type(json_str['jobs']) is dict and not json_str['multi_jobs']:
            json_jobs = [json_str['jobs']]
        
        elif type(json_str['jobs']) is list:
            json_jobs = json_str['jobs']

        for job in json_jobs:
            min_bool = False
            max_bool = False
            average_bool = False
            currency_bool = False
            j_name = list(job.keys())[0]
            dp = list(job.values())[0]
            salary_bool_temp = False
            if 'salary' in dp and type(dp['salary']) is dict:
                salary_bool_temp = True
                average_number = -1
                min_number = -1
                max_number = -1
                if 'salary_average' in dp['salary']:
                    average_number = judge_number(str(dp['salary']['salary_average']))
                    if average_number > 0:
                        average_bool = judge_average_in_sample(average_number, text)

                if 'salary_max' in dp['salary']:
                    max_number = judge_number(str(dp['salary']['salary_max']))
                    if max_number > 0:
                        max_bool = judge_maximal_in_sample(max_number, text)

                if 'salary_min' in dp['salary']:
                    min_number = judge_number(str(dp['salary']['salary_min']))
                    if min_number > 0:
                        min_bool = judge_minimal_without_k(min_number, text)

                if salary_bool_temp and 'currency' in dp['salary'] and type(dp['salary']['currency']) is str:
                    currency_bool = True


            if min_bool and max_bool and currency_bool:
                job_names.append(j_name)
                salary_bools.append(True)
                salary_values.append((min_number + max_number)/2)
                salary_currency.append(dp['salary']['currency'].lower())
                
            elif average_bool and currency_bool:
                job_names.append(j_name)
                salary_bools.append(True)
                salary_values.append(average_number)
                salary_currency.append(dp['salary']['currency'].lower())
                
            else:
                job_names.append(j_name)
                salary_bools.append(False)
                salary_values.append(-1)
                salary_currency.append('-1')

            


    job_dict = {'job_names': job_names,
                'salary_bools': salary_bools,
                'salary_values': salary_values,
                'salary_currency': salary_currency,}

    return job_dict


def detect_skill_str(json_str):
    job_names = []
    skill_lists = []
    
    if 'jobs' in json_str:
        if type(json_str['jobs']) is dict and json_str['multi_jobs']:
            json_jobs = [{jt:tt} for jt, tt in json_str['jobs'].items()]

        elif type(json_str['jobs']) is dict and not json_str['multi_jobs']:
            json_jobs = [json_str['jobs']]
        
        elif type(json_str['jobs']) is list:
            json_jobs = json_str['jobs']

        for job in json_jobs:
            j_name = list(job.keys())[0]
            dp = list(job.values())[0]
            skill_bool = False
            if 'skills' in dp and type(dp['skills']) is list:
                skill_bool = True
            
            if skill_bool:
                job_names.append(j_name)
                skill_lists.append([s for s in dp['skills']])
            else:
                job_names.append(j_name)
                skill_lists.append([])
                
                
    job_dict = {'job_names': job_names,
                'skill_lists': skill_lists}

    return job_dict


def check_job_city(dp, text):
    jstring = ''
    for s in dp:
        if s.isalpha():
            jstring += s.lower()

    textstring = ''
    for s in text:
        if s.isalpha():
            textstring += s.lower()

    if jstring in textstring:
        return True
    else:
        return False
    
def get_job_city_name(city_dict, job_str_dict):
    text = job_str_dict['title'] + job_str_dict['details']
    city_str = list(city_dict.values())[0]
    if type(city_str) is str:
        if check_job_city(city_str, text):
            return city_str
    elif type(city_str) is list:
        city_list_str = [city_str_temp for city_str_temp in city_str if check_job_city(city_str_temp, text)]
        temp = ''
        for c in sorted(city_list_str):
            temp += '___' + c
        return temp
    else:
        return -1




            