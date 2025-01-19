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
csv_files = os.listdir(f_table_result_folder_path)
# for each pair of csv files, compare the results, (A-B) and (B-A) is the same, only need to do one
ssub_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/RL-Model-for-Advise-Task/feng/uni_model/cluster_script/run_bms.ssub'
for i in range(len(csv_files)):
    for j in range(i+1, len(csv_files)):
        csv_file1 = csv_files[i]
        csv_file2 = csv_files[j]
        csv_file_path1 = f"{f_table_result_folder_path}/{csv_file1}"
        csv_file_path2 = f"{f_table_result_folder_path}/{csv_file2}"
        # if the two files are the same, skip
        if csv_file1 == csv_file2:
            continue
        stdout_name = f"{log_res_path}/{csv_file1}_{csv_file2}--%J.stdout"
        stderr_name = f"{log_res_path}/{csv_file1}_{csv_file2}--%J.stderr"
        job_name = f"bms_{csv_file1}_{csv_file2}"
        # submit the job
        command = f"sbatch -J {job_name} -o {stdout_name} -e {stderr_name} {ssub_path} {csv_file_path1} {csv_file_path2} {output_folder_path}"
        os.system(command)
        print(f"Submitted job {job_name} with command {command}")