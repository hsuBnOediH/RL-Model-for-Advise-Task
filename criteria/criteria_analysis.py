# read all the file from path: /mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice

import os
import pandas as pd
files = os.listdir('/mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice')

# read csv file contain all the subject ids
subject_list_path = '/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/subject_id/advise_subject_IDs_prolific_wo_uncomplete.csv'


subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'ID' not in line:
            subjects.append(line.strip())

# read each file of subject, the file named "active_trust_[subject_id]_T[1,2,3]_2024-07-25-02h29.56.847.csv"
# for each subject always read the latest and complete file
subject_files = {}
for subject in subjects:
    # find all the files name in the format "active_trust_[subject_id]_T[1,2,3]_2024-07-25-02h29.56.847.csv"
    temp_list = []
    for file in files:
        if file.startswith('active_trust_'+subject+'_T') and file.endswith('.csv'):
            temp_list.append(file)
    if len(temp_list) == 1:
        subject_files[subject] = temp_list[0]
    elif len(temp_list) > 1:
        print('--'*20)
        print('Multiple files for subject:', subject)
        # sort the files by date
        temp_list.sort()
        print('Files:', temp_list)
        for current_file in temp_list:
            # check if the file is complete by last row of file is start with 359 or not
            df = pd.read_csv('/mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice/'+current_file)
            if df.iloc[-1, 0] == 359:
                subject_files[subject] = current_file
                print(f"Selecting file {current_file} for subject {subject}")
                break
            else:
                print(f"File {current_file} for subject {subject} is incomplete")

    else:
        print('No file for subject:', subject)

print(f"Read {len(subject_files)} files")


# check for each subject, read as dataframe and save in dictionary
subject_data = {}
for subject, file in subject_files.items():
    df = pd.read_csv('/mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice/'+file)
    subject_data[subject] = df
print(f"Read {len(subject_data)} dataframes")

# for each dataframe, find the last row whose trial_type is MAIN, and remove rows before that, including that row
for subject, df in subject_data.items():
    last_main_row = df[df['trial_type'] == 'MAIN'].index[-1]
    subject_data[subject] = df.iloc[last_main_row+1:]



# check criteria 1
# for each subject, check event_type is 9, the 'result' is positive or negative
# postive means correct, negative means incorrect, count the number of correct and incorrect
criteria1 = {}
for subject, df in subject_data.items():
    correct = 0
    incorrect = 0
    for index, row in df.iterrows():
        if row['event_type'] == 9:
            if int(row['result']) > 0:
                correct += 1
            else:
                incorrect += 1
    criteria1[subject] = {'correct': correct, 'incorrect': incorrect}
print(criteria1)



# check criteria 2
# for each subject, check event_type is 8, the 'response' is left or right
# left means left, right means right, count the number of left and right
criteria2 = {}
for subject, df in subject_data.items():
    left = 0
    right = 0
    for index, row in df.iterrows():
        if row['event_type'] == 8:
            if row['response'] == 'left':
                left += 1
            elif row['response'] == 'right':
                right += 1
    criteria2[subject] = {'left': left, 'right': right}
print(criteria2)

