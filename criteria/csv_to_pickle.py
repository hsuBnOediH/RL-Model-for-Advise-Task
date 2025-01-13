# read all the file from path: /mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice
import os
import pandas as pd
import argparse

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
        # sort the files by date
        temp_list.sort()
        for current_file in temp_list:
            # check if the file is complete by last row of file is start with 359 or not
            df = pd.read_csv('/mnt/dell_storage/labs/NPC/DataSink/StimTool_Online/WB_Advice/'+current_file)
            if df.iloc[-1, 0] == 359:
                subject_files[subject] = current_file
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


# for each dataframe, find the last row whose trial_type is MAIN, and remove rows before that, including that row
for subject, df in subject_data.items():
    last_main_row = df[df['trial_type'] == 'MAIN'].index[-1]
    subject_data[subject] = df.iloc[last_main_row+1:]
print(f"Read {len(subject_data)} dataframes")

res_dict = {}
for subject, df in subject_data.items():
    temp_table = []
    header =["idx", ",advice", "action","reward"] 
    for i in range(360):
        temp_df = df[df['trial'] == i]
        if len(temp_df) == 0:
            temp_table.append([i, 'NA', 'NA', 'NA'])
        else:
            advice = 'NA'
            if len(temp_df[temp_df['event_type'] == 9]) > 1:
                advice = temp_df[temp_df['event_type'] == 9].iloc[0]['result']
                advice = advice.split(' ')[1]
            action = temp_df[temp_df['event_type'] == 8].iloc[0]['response']
            reward = temp_df[temp_df['event_type'] == 9].iloc[-1]['result']
            reward = int(reward)
            temp_table.append([i, advice, action, reward])
    res_dict[subject] = temp_table

# save the dictionary to pickle file
import datetime
import pickle
file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'_subject_data.pkl'
with open(file_name, 'wb') as handle:
    pickle.dump(res_dict, handle)
print(f"Save the data to {file_name}")



            
