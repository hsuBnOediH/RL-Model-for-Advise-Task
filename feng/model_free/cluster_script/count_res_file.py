import sys, os, re, subprocess

import argparse

import copy



parser = argparse.ArgumentParser(description="Process a list of numbers or ranges.")

parser.add_argument("idx", type=int, help="Specify the index of the candidate.", default=0)



args = parser.parse_args()

idx_candidate = args.idx



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

if idx_candidate == 0:

    idx_candidate = 10

for model_idx in range(1, idx_candidate+1):

    total_subjects = len(subjects)

    connected_subjects = 0

    print(f"model {model_idx}:")

    non_res_connected_subjects = copy.deepcopy(subjects)

    for subject in subjects:

        # FIT_SUBJECT, '_candidate_', num2str(IDX_CANDIDATE), is_connedted_str, '.csv'])

        file_name = subject + '_candidate_' + str(model_idx) + '_connected' + '.csv'

        if file_name in files:

            connected_subjects += 1

            non_res_connected_subjects.remove(subject)

    print(f"model {model_idx}: connected : {connected_subjects}/{total_subjects}")

    #print(f"Non-res connected subjects: {non_res_connected_subjects}")



    total_subjects = len(subjects)

    disconnected_subjects = 0

    non_res_disconnected_subjects = copy.deepcopy(subjects)

    for subject in subjects:

        file_name = subject + '_candidate_' + str(model_idx) + '_disconnected' + '.csv'

        if file_name in files:

            disconnected_subjects += 1

            non_res_disconnected_subjects.remove(subject)

    print(f"model {model_idx}: disconnected : {disconnected_subjects}/{total_subjects}")

    print("-"*20)

    #print(f"Non-res disconnected subjects: {non_res_disconnected_subjects}")

