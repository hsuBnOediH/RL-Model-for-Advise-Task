import sys, os, re, subprocess

f_table_result_folder_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/compare_result'
import datetime
res_folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ '_run'
output_folder_path = f_table_result_folder_path + '/' + res_folder_name
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    print(f"Created output folder {output_folder_path}")
log_res_path = f"{output_folder_path}/logs"
# create those folders

if not os.path.exists(log_res_path):
    os.makedirs(log_res_path)
    print(f"Created logs folder {log_res_path}")

# list all the csv files in the folder
csv_files = []
all_files = os.listdir(f_table_result_folder_path)
for file in all_files:
    if file.endswith('.csv'):
        csv_files.append(file)

# for each pair of csv files, compare the results, (A-B) and (B-A) is the same, only need to do one
ssub_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/RL-Model-for-Advise-Task/feng/uni_model/cluster_script/run_bms.ssub'
for i in range(len(csv_files)):
    for j in range(i+1, len(csv_files)):
        csv_file1 = csv_files[i]
        csv_file2 = csv_files[j]
        cvs_file1_wo_ext = csv_file1[:-4]
        cvs_file2_wo_ext = csv_file2[:-4]

        csv_file_path1 = f"{f_table_result_folder_path}/{csv_file1}"
        csv_file_path2 = f"{f_table_result_folder_path}/{csv_file2}"
        # if the two files are the same, skip
        if csv_file1 == csv_file2:
            continue
        stdout_name = f"{log_res_path}/{cvs_file1_wo_ext}_{cvs_file2_wo_ext}--%J.stdout"
        stderr_name = f"{log_res_path}/{cvs_file1_wo_ext}_{cvs_file2_wo_ext}--%J.stderr"
        job_name = f"bms_{cvs_file1_wo_ext}_{cvs_file2_wo_ext}"
        # submit the job
  

        model_1_name = cvs_file1_wo_ext.split('-')[0]
        model_2_name = cvs_file2_wo_ext.split('-')[0]
        author_1_name = cvs_file1_wo_ext.split('-')[1].split('_')[0]
        author_2_name = cvs_file2_wo_ext.split('-')[1].split('_')[0]

        print(f"job_name: {job_name}")
        print(f"stdout_name: {stdout_name}")
        print(f"stderr_name: {stderr_name}")
        print(f"ssub_path: {ssub_path}")
        print(f"csv_file_path1: {csv_file_path1}")
        print(f"csv_file_path2: {csv_file_path2}")
        print(f"output_folder_path: {output_folder_path}")
        print(f"model_1_name: {model_1_name}")
        print(f"model_2_name: {model_2_name}")
        print(f"author_1_name: {author_1_name}")
        print(f"author_2_name: {author_2_name}")
        



        command = f"sbatch -J {job_name} -o {stdout_name} -e {stderr_name} {ssub_path} {csv_file_path1} {csv_file_path2} {model_1_name} {model_2_name} {author_1_name} {author_2_name} {output_folder_path}"
        os.system(command)
        print(f"Submitted job {job_name}")