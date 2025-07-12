# The task space analysis

This is the code for paper https://arxiv.org/abs/2504.03581

Right now this manuscript is submitted and the dataset link is only available for reviewers and editors. The link for dataset is provided in the submitted manuscript. The dataset will be public available once the manuscript got accepted.

# Before runing the programs:

Unzip the "task_space_data.tar.gz" and put the data folder "task_space_data" into the "task_space_build_analysis/data_processing/data_files/"

# Get the figures:

In the "task_space_build_analysis", there are a bunch of Jupyter Notebook. Each notebook will generate the corresponding figure for the paper. I keep the necessary data to generate the figures in the dataset file.

# To rebuild the whold dataset (it takes several days)

If you want to rebuild everything and run the code from the very beginning raw data, there is a "main_data_processing.py" file in the "task_space_build_analysis/data_processing/". 

Or you can run the Jupyter Notebooks in the directory "task_space_build_analysis/data_processing/". I order the Jupyter Notebooks by numbers. You can run it one by one. 

The data processing and rebuilding is very time-consuming. It will take several days to finish running all the code.

There are some parts involving api-keys and random functions. I comment them right now since they may generate different outputs. I will update them in the future.

# Environment Requirement

The package required for Python is in the "environment.yaml".

The required R packages are bipartite (version 2.21), fixest (version 0.12.1), stargazer (version 5.2.3), dplyr (version 1.0.10), mefa4 (version 0.3-9), data.table (version 1.17.8).

Dependent on the Interset quality, it usually takes short time to install all the packages.

We run the code on Linux or Windows 11. 

# Hardware Requirement

A GPU is required to run all the code.

## License

This project is released under the [Creative Commons Attribution 4.0 International License](LICENSE).
