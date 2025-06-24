import dropbox
import os

CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunk size

def upload_large_file(dbx, local_path, dropbox_path):
    with open(local_path, "rb") as f:
        file_size = os.path.getsize(local_path)
        if file_size <= CHUNK_SIZE:
            dbx.files_upload(f.read(), dropbox_path)
        else:
            upload_session_start_result = dbx.files_upload_session_start(f.read(CHUNK_SIZE))
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                    offset=f.tell())
            while f.tell() < file_size:
                if (file_size - f.tell()) <= CHUNK_SIZE:
                    dbx.files_upload_session_finish(f.read(CHUNK_SIZE),
                                                    cursor,
                                                    dropbox.files.CommitInfo(path=dropbox_path))
                else:
                    dbx.files_upload_session_append_v2(f.read(CHUNK_SIZE), cursor)
                    cursor.offset = f.tell()



dbx = dropbox.Dropbox("sl.u.AFbgzXe5O5O9FwPOHgb3S57dK80VsnnOKeM3oXTvErBcRo9k2WlDXvE4lFxULJGf7Dj073MChuI_qsait4q2ukWp2Hp8NcAIVA7L6SI7sRkT9BCIixQb-vZN5kci9uR-edS1aqFeX9GFTn5QwjqIsSqCPa5EXbxmIzq4d2m7tgYsJEb_kcI5RMieX4m9ZDfi1bCW6IeTL2Dqy7Y8LxfEOJSadubZIki8oLzJweHXNzEY8ACPQJhpSBgl2BPEf0p3D4VQH44m0LzUKP0tHvj3yO1CzgDoOeqKKQeX3n4CTAcuuXf5wNLThN0xlP4OO1ysZ49NLoL5llFPgCo5puh2oOafuKVkTV9mvTyfafRq3COjhV3OzpqPgmsUopfPUBaGr1FRyCWbzHnlMy1fSWznWSuVEuY5edVFDDS6YMt9UkIpJLEIk6IepKcAT9w2YjAs_plOOLsLOoznowzFKpcZ_sKXCYJ5QW9kyPl-JSaGtCK2tpC0bibJ3f-bMmHzlieeGChjikiliUOc3t9GeqPfepyw58nc2h5d4-N7ScNrR4sULD-Y-t98gCO0FcMSiBG87at9URAscivJL8Asf1HW3CJf4ehcRI-zzZI_sNP36YUHvn24T_PZsBvFNNpjzv-okDXxOGynnfqXMAA5M_aXTQVzXgFTXjbYc0GzfxwsmWFPZS-UB8aP5PrgFcqkOIv1gXP5aXZ-pIs1mXZY7zyiMFvwbBabiP1kCaG8heciE-raK92chzrtoNZY2HmENjIGsb2KdPw9G-H4ZziLxEh-xNygyQydnwZDrJkZeWOxHnNA3Nqv8kaxZs3KXRTUnA6te6i8btT5msZTWX1Mv8Nry7mC0idiBVRiYIJVYrofeqFNm1nJR-at9dP8FPsRJSoj0PSZZxxVn96GgJjnGcH0VYN1ew7zG7U6hqlIhwWYutS-0fT-4TNKuPDpaolBrsjOcPkyXCmJH_3HrIN0m4KuDSfxlURsEefG9iVTuBoh6f6DsZxoMbHe4crJrNLcs9y01ODdifSscrDcWqNVweCYh-PYMUn_1Hk0qW6-w-0EZ4soD4cmAvYL0G-cD-YqRFbsQ-kF757eg45ZPsQMcK7EHsfBxHmvucmxOyo7Dodsae9T9psd9ZMRKDPpZZP-LUfv8Yr2uHRltZR3z5Bl_Oa_6Io9ItV_uXbcwBAZ4JxvANxbZ_FkX6LWkjqMKqbcYoBIniLvoN5kq1Ff3CF7AE8jB-wwBnwCc14edTy1YosbKXUn6TcIIACtT7V3xzfRq6qsYG1I7mUMe9HgSDLI_DCKcL2m")

data_path = '/home/xiangnan/task_space_code/task_space_data/'
#data_path = 'D:/CSH/stack_overflow/task_space_data/'
data_path_save = data_path + 'obj_tag_question_bipartite_core_space/'

file_name = data_path_save + 'vote_regression_together/user_task_collection/df_sample/' + 'df_all_language_10percent.zip'

upload_large_file(dbx, file_name, "Apps/user_task_dataframe/")

