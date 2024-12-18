import sys, os, re, subprocess
import argparse
import copy
subject_list_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/advise_subject_IDs_prolific_wo_uncomplete.csv'

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())

res_parent_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/'
# find all the subfolders in the parent folder
subfolders = [f.path for f in os.scandir(res_parent_path) if f.is_dir()]
# the forat datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+ '_run', find the latest one
subfolders.sort()
latest_folder = subfolders[-1]
res_folder = latest_folder+'/temp_res'
# find all the files in the folder
files = os.listdir(res_folder)

# format is  output_file = fullfile(RES_PATH, [FIT_SUBJECT, '_candidate_', num2str(IDX_CANDIDATE), is_connedted_str, '.csv']);
# find the file name that contain the subject id

total_subjects = len(subjects)
connected_subjects = 0
non_res_connected_subjects = copy.deepcopy(subjects)
for subject in subjects:
    if any([subject in file and 'connected' in file for file in files]):
        connected_subjects += 1
        non_res_connected_subjects.remove(subject)
print(f"Total subjects: {total_subjects}, connected subjects: {connected_subjects}")
print(f"Non-res connected subjects: {non_res_connected_subjects}")

total_subjects = len(subjects)
disconnected_subjects = 0
non_res_disconnected_subjects = copy.deepcopy(subjects)
for subject in subjects:
    if any([subject in file and 'disconnected' in file for file in files]):
        continue
print(f"Total subjects: {total_subjects}, disconnected subjects: {disconnected_subjects}")
print(f"Non-res disconnected subjects: {non_res_disconnected_subjects}")
