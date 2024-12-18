import sys, os, re, subprocess
import argparse
subject_list_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/advise_subject_IDs_prolific_wo_uncomplete.csv'
debug_subject_list_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/debug_subjects.csv'
# use current time generate a folder to save the results
import datetime
folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ '_run'
output_folder_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/'+folder_name
# if folder not exist, create it
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    print(f"Created output folder {output_folder_path}")

SIM=False
FIT=True
PLOT=False

temp_res_path = f"{output_folder_path}/temp_res"
log_res_path = f"{output_folder_path}/logs"
final_res_path = f"{output_folder_path}/final_res"
# create those folders
if not os.path.exists(temp_res_path):
    os.makedirs(temp_res_path)
    print(f"Created temp_res folder {temp_res_path}")
if not os.path.exists(log_res_path):
    os.makedirs(log_res_path)
    print(f"Created logs folder {log_res_path}")
if not os.path.exists(final_res_path):
    os.makedirs
    print(f"Created final_res folder {final_res_path}")


parser = argparse.ArgumentParser(description="Process a list of numbers or ranges.")

parser.add_argument("range", type=str, help="Specify numbers as a single number (e.g., '1') or a range (e.g., '1-4').",default='1-10')
# need a arg for connected version or not, use 
parser.add_argument("connected", type=int, help="Specify if the connected version is used.", default=0)
# need a arg for if the debug subject list is used, use -d or --debug, default is False
parser.add_argument("-d", "--debug", action="store_true", help="Use debug subject list.", default=False)

args = parser.parse_args()
range_str = args.range
is_connected = args.connected
if is_connected == 0:
    is_connected = 'False'
elif is_connected == 1:
    is_connected = 'True'
else:
    is_connected = 'Both'

if args.debug:
    subject_list_path = debug_subject_list_path
number_list = []
if '-' in range_str:
    start, end = map(int, range_str.split('-'))
    number_list = list(range(start, end + 1))
else:
    number_list = [int(range_str)]



subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())

ssub_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/RL-Model-for-Advise-Task/feng/model_free/cluster_script/run_model_free.ssub'

print(f"subject_file_path: {subject_list_path}")
print(f"temp_res_path: {temp_res_path}")
print(f"idx_candidate: {number_list}")
print(f"is_connected: {is_connected}")


if is_connected == 'Both':
    for is_connected in ['True', 'False']:
        for idx_candidate in number_list:
            for subject in subjects:
                stdout_name = f"{log_res_path}/{subject}-{idx_candidate}-{is_connected}-%J.stdout"
                stderr_name = f"{log_res_path}/{subject}-{idx_candidate}-{is_connected}-%J.stderr"
                jobname = f'rl-advise-comparison-{subject}-{idx_candidate}-{is_connected}'
                os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {SIM} {FIT} {subject} {temp_res_path} {idx_candidate} {is_connected} {PLOT}")
                print(f"SUBMITTED JOB [{jobname}]")
else:
    for idx_candidate in number_list:
        for subject in subjects:
            stdout_name = f"{log_res_path}/{subject}-{idx_candidate}-{is_connected}-%J.stdout"
            stderr_name = f"{log_res_path}/{subject}-{idx_candidate}-{is_connected}-%J.stderr"
            jobname = f'rl-advise-comparison-{subject}-{idx_candidate}-{is_connected}'
            os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {SIM} {FIT} {subject} {temp_res_path} {idx_candidate} {is_connected} {PLOT}")
            print(f"SUBMITTED JOB [{jobname}]")
