import os

# find the path of all the csv files in the folder

folder_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/2024-12-19-11-43-17_run/temp_res'

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

a_inf_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/results/model_comparison/final_res/output_f_values_comparison.csv'

# read the file from the second row, the first cloumn is the subject id, each cloumn is the f value of a model

a_inf_table = []

with open(a_inf_path) as infile:

    for line_idx, line in enumerate(infile):

        if line_idx == 0:

            res_table[0].extend(line.strip().split(',')[1:])

        else:

            row = line.strip().split(',')

            a_inf_subject = row[0]

            if a_inf_subject == res_table[line_idx][0]:

                res_table[line_idx].extend(row[1:])

                print(f"subject {a_inf_subject} added to the table, row {line_idx},progress {line_idx/len(res_dict)}")

            else:

                print(f"subject {a_inf_subject} not in the model free table")

                break









# write the table into a csv file

import csv

output_path = f'/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/rl_results/RL/model_free/{run_name}/F_table.csv'

with open(output_path, 'w', newline='') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerows(res_table)

print(f"Write the F table to {output_path}")



       