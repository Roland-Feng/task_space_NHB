library(bipartite)

discrepancy_list <- list()
discrepancy2_list <- list()
binmatnest_list <- list()
NODF_list <- list()
NODF2_list <- list()
checker_list <-list()


file_loc =  'data_files/task_space_data/obj_tag_question_bipartite_core_space/task_language_nestedness/tl_matrix/'
file_name <-'answer_language_task_matrix_all_year.csv'

file_matrix <- paste(file_loc, file_name, sep='')
m<-read.csv(file_matrix, header=FALSE)

discrepancy_list <- append(discrepancy_list, nested(m, method = 'discrepancy'))
discrepancy2_list <- append(discrepancy2_list, nested(m, method = 'discrepancy2'))
binmatnest_list<-append(binmatnest_list, nested(m, method = 'binmatnest'))
NODF_list <- append(NODF_list, nested(m, method = 'NODF'))
NODF2_list <- append(NODF2_list, nested(m, method = 'NODF2'))
checker_list <-append(checker_list, nested(m, method = 'checker'))

discrepancy_matrix <- matrix(discrepancy_list)
discrepancy2_matrix <- matrix(discrepancy2_list)
binmatnest_matrix<-matrix(binmatnest_list)
NODF_matrix <- matrix(NODF_list)
NODF2_matrix <- matrix(NODF2_list)
checker_matrix <-matrix(checker_list)

write.csv(discrepancy_matrix, file=paste(file_loc,'discrepancy_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
write.csv(discrepancy2_matrix, file=paste(file_loc,'discrepancy2_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
write.csv(binmatnest_matrix, file=paste(file_loc,'binmatnest_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
write.csv(NODF_matrix, file=paste(file_loc,'NODF_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
write.csv(NODF2_matrix, file=paste(file_loc,'NODF2_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
write.csv(checker_matrix, file=paste(file_loc,'checker_matrix_all_year.csv', sep = ''), row.names=FALSE, col.names=FALSE)
