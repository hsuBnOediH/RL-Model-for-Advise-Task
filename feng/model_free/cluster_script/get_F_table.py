import os
import argparse
# find the path of all the csv files in the folder
parent_folder = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free'

parser = argparse.ArgumentParser(description="Process a list of numbers or ranges.")
parser.add_argument("run_folder_idx", type=int, help="Specify the index of the run folder.", default=1)
parser.add_argument("active_inference_idx", type=int, help="Specify the index of the active inference model.", default=0)
args = parser.parse_args()
run_folder_idx = args.run_folder_idx
active_inference_idx = args.active_inference_idx

# list all the folders in the parent folder and sort them by name, all the folders are named in format like 2024-12-19-11-43-17_run
folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
folders.sort()
folder_path = folders[-run_folder_idx] + '/temp_res'

# folder_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/2024-12-19-11-43-17_run/temp_res'
run_name = folder_path.split('/')[-2]
files = os.listdir(folder_path)
print(len(files))

res_dict = {}

for file in files:

    if file.endswith('.csv'):

        # remove the .csv

        file_name_no_ext = file[:-4]

        file_split = file_name_no_ext.split('_')

        subject_id = file_split[0]

        model_idx = int(file_split[2])

        connected = file_split[3]

        if subject_id not in res_dict:

            res_dict[subject_id] = {}

        if model_idx not in res_dict[subject_id]:

            res_dict[subject_id][model_idx] = {}

        res_dict[subject_id][model_idx][connected] = file



# write the value of dict into a 2 D array, each row is a subject, each column is a model


res_table = []

# write the header

header = ['subject']

for model_idx in range(1, 11):

    for connected in ['connected', 'disconnected']:

        header.append(f"model_{model_idx}_{connected}")

res_table.append(header)

subject_idx = 0



# sort the subject by the subject id

subjects = list(res_dict.keys())

subjects.sort()

for subject in subjects:

    row = []

    row.append(subject)

    for model_idx in range(1, 11):

        for connected in ['connected', 'disconnected']:

            file = res_dict[subject].get(model_idx, {}).get(connected, None)

            # read the last cell of the second row

            f_value = None

            if file is not None:

                with open(os.path.join(folder_path, file)) as infile:

                    for line in infile:

                        pass

                    last_line = line.strip()

                    # split the last line by ',' and get the last value transformed into float

                    f_value = float(last_line.split(',')[-1])

            row.append(f_value)

    res_table.append(row)

    subject_idx += 1

    progress = subject_idx / len(res_dict)

    print(f"subject {subject}, row {subject_idx}, progress {progress}")



# append Active Inference model result into the table
a_inf_parent_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/ai_results/'
a_inf_subfolders = os.listdir(a_inf_parent_path)
a_inf_subfolders.sort()
if active_inference_idx == 0:
    a_inf_path = None
else:
    a_inf_path = a_inf_parent_path + a_inf_subfolders[-active_inference_idx] + '/temp_res'


if a_inf_path is not None:

    # add all the model idx and subject id into the table header
    for model_idx in range(1, 11):
        header.append(f"active_inference_{model_idx}")
    res_table[0] = header

    a_inf_files = os.listdir(a_inf_path)
    # for loop the model idx and subject id to find all the F value
    for model_idx in range(1, 11):
        for subject in subjects:
            # the file name need to change to 
            file = subject + '_candidate_' + str(model_idx) + '.csv'
            if file in a_inf_files:
                with open(os.path.join(a_inf_path, file)) as infile:
                    for line in infile:
                        pass
                    last_line = line.strip()
                    f_value = float(last_line.split(',')[-1])
                res_table[subjects.index(subject)+1].append(f_value)
            else:
                res_table[subjects.index(subject)+1].append(None)


# exclude the subject id by reading file 
excluded_subjects_file_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/excluded_ids.csv'
excluded_subjects_ids = []
with open(excluded_subjects_file_path) as infile:
    for line in infile:
        excluded_subjects_ids.append(line.strip())

for subject_id in excluded_subjects_ids:
    if subject_id in subjects:
        res_table.pop(subjects.index(subject_id)+1)
        subjects.remove(subject_id)


# write the table into a csv file
import csv

output_path = f'/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/{run_name}/F_table.csv'

with open(output_path, 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerows(res_table)

print(f"Write the F table to {output_path}")



       