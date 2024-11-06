import sys, os, re, subprocess
import argparse
subject_list_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/advise_subject_IDs_prolific_wo_uncomplete.csv'
results = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/results/RL/model_free/'

SIM=False
FIT=True
PLOT=False

temp_res_path = f"{results}/temp_res"

if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")


if not os.path.exists(f"{results}/final_res"):
    os.makedirs(f"{results}/final_res")
    print(f"Created results-logs directory {results}/final_res")

if not os.path.exists(f"{results}/temp_res"):
    os.makedirs(f"{results}/temp_res")
    print(f"Created results-logs directory {results}/temp_res")

parser = argparse.ArgumentParser(description="Process a list of numbers or ranges.")
parser.add_argument("range", type=str, help="Specify numbers as a single number (e.g., '1') or a range (e.g., '1-4').")

args = parser.parse_args()
range_str = args.range
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

for idx_candidate in number_list:
    for subject in subjects:
        stdout_name = f"{results}/logs/{subject}-%J.stdout"
        stderr_name = f"{results}/logs/{subject}-%J.stderr"
        jobname = f'advise-comparison-{subject}-{idx_candidate}'
        os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {SIM} {FIT} {PLOT} {subject} {temp_res_path} {idx_candidate}")

        print(f"SUBMITTED JOB [{jobname}]")