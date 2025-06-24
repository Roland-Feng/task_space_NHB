library(fixest)
library(stargazer)
library(dplyr)
library(mefa4)
library(data.table)

file_dir <- 'data_files/task_space_data/obj_tag_question_bipartite_core_space/vote_regression_together/user_task_collection/df_sample/'


file_all_language <- 'df_user_task_entry_exit_salary_task_with_density_level_1_10_percent_all_language.csv'
fdf = fread(paste(file_dir, file_all_language, sep = ''))
fdf <- subset(fdf, entry_sign != 2)

fdf$task_value_centered <- (fdf$task_value - mean(fdf$task_value, na.rm = TRUE))
fdf$density_2008_2023_centered <- (fdf$density_2008_2023 - mean(fdf$density_2008_2023, na.rm = TRUE)) / sd(fdf$density_2008_2023, na.rm = TRUE)


m1 <- feols(entry_sign ~ task_value_centered, cluster = ~user_id, data=subset(fdf))

m2 <- feols(entry_sign ~ task_value_centered |  user_id, cluster = ~user_id, data=subset(fdf))

m3 <- feols(entry_sign ~ task_value_centered + density_2008_2023_centered , cluster = ~user_id, data=subset(fdf))

m4 <- feols(entry_sign ~ task_value_centered + density_2008_2023_centered |  user_id, cluster = ~user_id, data=subset(fdf))

etable(m1,m2, m3,m4,drop = 'Constant', tex = FALSE)

confint(m1)
confint(m2)
confint(m3)
confint(m4)
