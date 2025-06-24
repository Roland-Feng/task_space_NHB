# task_space_NHB

This is the code for paper https://arxiv.org/abs/2504.03581

Right now this manuscript is submitted and the dataset link is only available for reviewers and editors. The dataset will be public available once the manuscript got accepted.

# Before runing the programs:

Unzip the "task_space_data.tar.gz" and put the data folder "task_space_data" into the "task_space_build_analysis/data_processing/data_files/"

# Get the figures:

In the "task_space_build_analysis", there are a bunch of Jupyter Notebook. Each notebook will generate the corresponding figure for the paper. I keep the necessary data to generate the figures in the dataset file.

# To rebuild the whold dataset (it takes several days)

If you want to rebuild everything and run the code from the very beginning raw data, there is a "main_data_processing.py" file in the "task_space_build_analysis/data_processing/". 

Or you can run the Jupyter Notebooks in the directory "task_space_build_analysis/data_processing/". I order the Jupyter Notebooks by numbers. You can run it one by one. 

The data processing and rebuilding is very time-consuming. It will take several days to finish running all the code.

There are some parts involving api-keys and random functions. I comment them right now since they may generate different outputs. I will update them in the future.
