library(fixest)
library(stargazer)
library(dplyr)
library(mefa4)
library(data.table)

file_dir <- 'data_files/task_space_data/obj_tag_question_bipartite_core_space/vote_regression_together/df_task_IV/'
file_name <- 'df_vote_task_exp_IV_all_threshold_user_level_1_all_year_period_2.csv'

fdf = fread(paste(file_dir, file_name, sep = ''))
#fdf <- subset(fdf, parent_ans_number > 1)


m1<-feols(answer_top_all_users  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1) | year, data=subset(fdf))

m2<-feols(answer_top_all_users  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1)  | group_task_year, data=subset(fdf))

m3<-feols(log(answer_vote + 1)  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1) | year, data=subset(fdf))

m4<-feols(log(answer_vote + 1)  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1)  |group_task_year, data=subset(fdf))

etable(m1,m2,m3,m4,drop = 'Constant', order = c("task_experience", "parent_ans_number", "all_answer_vote"), tex = TRUE)

confint(m1)
confint(m2)
confint(m3)
confint(m4)


fdf2 <- subset(fdf, parent_ans_number > 1)


m1<-feols(answer_top_all_users  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1) | year, data=subset(fdf2))

m2<-feols(answer_top_all_users  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1)  | group_task_year, data=subset(fdf2))

m3<-feols(log(answer_vote + 1)  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1) | year, data=subset(fdf2))

m4<-feols(log(answer_vote + 1)  ~ log(task_experience + 1) + log(parent_ans_number + 1) + log(all_answer_vote + 1)  |group_task_year, data=subset(fdf2))

etable(m1,m2,m3,m4,drop = 'Constant', order = c("task_experience", "parent_ans_number", "all_answer_vote"), tex = TRUE)


## taskiv1
m1<-feols(answer_top_all_users ~ log(parent_ans_number + 1) +  log(all_answer_vote + 1)  | year + q_absminute |log(task_experience + 1) ~ task_IV_log, data=subset(fdf), cluster = ~group_task_qminute)

## taskiv2
m2<-feols(answer_top_all_users ~ log(parent_ans_number + 1) +  log(all_answer_vote + 1) | group_task_year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf), cluster = ~group_task_qminute)

## votestaskiv1
m3<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf), cluster = ~group_task_qminute)

## votestaskiv2
m4<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | q_absminute + group_task_year|log(task_experience + 1) ~ task_IV_log, data=subset(fdf), cluster = ~group_task_qminute)

etable(m1,m2,m3,m4,drop = 'Constant', order = c("task_experience", "parent_ans_number", "all_answer_vote"), tex = TRUE)




## placeboiv4
m1<-feols(answer_top_all_users ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute < 1440), cluster = ~group_task_qminute)

## placeboiv5
m2<-feols(answer_top_all_users ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | q_absminute + group_task_year|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute < 1440), cluster = ~group_task_qminute)

## vplaceboiv4
m3<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute<1440), cluster = ~group_task_qminute)

## vplaceboiv5
m4<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | q_absminute + group_task_year|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute<1440), cluster = ~group_task_qminute)


## placeboiv1
m5<-feols(answer_top_all_users ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute>1440), cluster = ~group_task_qminute)

## placeboiv2
m6<-feols(answer_top_all_users ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | q_absminute + group_task_year|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute > 1440), cluster = ~group_task_qminute)

## vplaceboiv1
m7<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | year + q_absminute|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute>1440), cluster = ~group_task_qminute)

## vplaceboiv2
m8<-feols(log(answer_vote + 1) ~ log(parent_ans_number + 1) + log(all_answer_vote + 1) | q_absminute + group_task_year|log(task_experience + 1) ~ task_IV_log, data=subset(fdf, aq_minute>1440), cluster = ~group_task_qminute)

etable(m1,m2,m3,m4,m5,m6,m7,m8,drop = 'Constant', order = c("task_experience", "parent_ans_number", "all_answer_vote"), tex = TRUE)
