import os
import argparse
from tqdm import tqdm
# find the path of all the csv files in the folder
parent_folder = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/uni_model/'

parser = argparse.ArgumentParser(description="Process a list of numbers or ranges.")
parser.add_argument("run_folder_idx", type=int, help="Specify the index of the run folder.", default=1)
args = parser.parse_args()

run_folder_idx = args.run_folder_idx

# list all the folders in the parent folder and sort them by name, all the folders are named in format like 2024-12-19-11-43-17_run
folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
folders.sort()
folder_path = folders[-run_folder_idx] + '/temp_res'

model_string = None

# folder_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/2024-12-19-11-43-17_run/temp_res'
run_name = folder_path.split('/')[-2]
files = [[],[],[],[],[]]
for params_idx in range(1, 6):
    temp_folder_path = folder_path + f'/paramcombi{params_idx}'
    temp_files = os.listdir(temp_folder_path)
    # find all the csv files in the folder
    for file in temp_files:
        if file.endswith('.csv'):
            files[params_idx-1].append(file)
print(f"total files number: {sum([len(files[i]) for i in range(5)])}")
res_dict = {}

for params_idx in range(1, 6):
    for file in files[params_idx-1]:
        if file.endswith('.csv'):
            file_name_no_ext = file[:-4]
            file_split = file_name_no_ext.split('-')
            file_split = file_split[1].split('_')
            subject_id = file_split[0]

            if subject_id not in res_dict:
                res_dict[subject_id] = [None,None,None,None,None]
            res_dict[subject_id][params_idx-1] = file

res_table = []
# sort the subject by the subject id
header = ['subject']
header_flag = False


res_table.append(header)

subject_idx = 0
# sort the subject by the subject id
subjects = list(res_dict.keys())
subjects.sort()

with tqdm(total=len(res_dict)) as pbar:
    for subject in subjects:
        row = [subject]
        for params_idx in range(1, 6):
            params_path = folder_path + f'/paramcombi{params_idx}'
            file = res_dict[subject][params_idx-1]
            with open(os.path.join(params_path, file)) as infile:
                for line in infile:
                    pass
                last_line = line.strip()
                last_line_split = last_line.split(',')
                f_value = float(last_line_split[-2])
                model_type = int(last_line_split[-1])
            if not header_flag:

                if model_type == 1:
                    model_string = 'active_inference_'
                elif model_type == 2:
                    model_string = 'rl_connected_'
                elif model_type == 3:
                    model_string = 'rl_disconnected_'
                for param_idx in range(1, 6):
                    header.append(model_string + str(param_idx))
                header_flag = True
            
            row.append(f_value)
        res_table.append(row)
        
        subject_idx += 1
        pbar.update(1)


excluded_subjects_file_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/excluded_ids.csv'
excluded_subjects_ids = []
with open(excluded_subjects_file_path) as infile:
    for line in infile:
        excluded_subjects_ids.append(line.strip())

for subject_id in excluded_subjects_ids:
    if subject_id in subjects:
        res_table.pop(subjects.index(subject_id)+1)
        subjects.remove(subject_id)
print(f"Total subjects: {len(subjects)}, excluded subjects: {len(excluded_subjects_ids)}")

# write the table into a csv file
import csv
output_path = f'/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/uni_model/{run_name}/{model_string}F_table.csv'
# create the output file and write the table into it, if the file already exists, overwrite it


with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(res_table)
print(f"Write the F table to {output_path}")



       