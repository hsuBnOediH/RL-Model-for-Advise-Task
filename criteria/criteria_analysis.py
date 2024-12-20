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



# check criteria 1
# for each subject, check event_type is 9, the 'result' is positive or negative
# postive means correct, negative means incorrect, count the number of correct and incorrect
criteria1 = {}
for subject, df in subject_data.items():
    correct = 0
    incorrect = 0
    for index, row in df.iterrows():
        if row['event_type'] == 9:
            if row['result'] in ('try left', 'try right'):
                continue
            if int(row['result']) > 0:
                correct += 1
            else:
                incorrect += 1
    criteria1[subject] = {'correct': correct, 'incorrect': incorrect}
print(f"finished criteria1")



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
print(f"finished criteria2")


# check criteria 3
# reaction time defined as the time between event_type 8 and 5
# for each subject, calculate the average reaction time
criteria3 = {}
for subject, df in subject_data.items():
    reaction_times = []
    total_reaction_time = 0
    for i in range(360):
       # group by trial
        temp_df = df[df['trial'] == i]
        if len(temp_df) == 0:
            continue
        # check if took advice by how many times event_type 5, 1 times means not took advice,2 times means took advice
        if len(temp_df[temp_df['event_type'] == 5]) == 1:
            #  evnt_type 8 absolute time - event_type 5 absolute time
            for index, row in temp_df.iterrows():
                if row['event_type'] == 5:
                    first_stim = row['absolute_time']
                elif row['event_type'] == 8:
                    first_action = row['absolute_time']
            reaction_times.append(first_action - first_stim)
            total_reaction_time += first_action - first_stim
        else:
            reaction_time = 0
            find_first_stim = False
            first_stim,first_stim,second_stim,second_action = None,None,None,None
            for index, row in temp_df.iterrows():
                if row['event_type'] == 5 and not find_first_stim:
                    first_stim = row['absolute_time']
                    find_first_stim = True
                elif row['event_type'] == 6:
                    first_action = row['absolute_time']
                elif row['event_type'] == 8:
                    second_action = row['absolute_time']
                elif row['event_type'] == 5 and find_first_stim:
                    second_stim = row['absolute_time']
                   
            if first_stim is None or first_action is None or second_stim is None or second_action is None:
                print(temp_df)
                print(f"first_stim: {first_stim}, first_action: {first_action}, second_stim: {second_stim}, second_action: {second_action}")
                break
            else:
                total_reaction_time += (first_action - first_stim) + (second_action - second_stim)
                reaction_times.append(((first_action - first_stim), (second_action - second_stim)))
    criteria3[subject] = {'average_reaction_time': total_reaction_time/len(reaction_times), 'reaction_times': reaction_times}

print(f"finished criteria3")



# check criteria 4
# for each subject, calculate the number of trials that took advice
criteria4 = {}
for subject, df in subject_data.items():
    took_advice = 0
    for i in range(360):
        temp_df = df[df['trial'] == i]
        if len(temp_df) == 0:
            continue
        if len(temp_df[temp_df['event_type'] == 5]) == 2:
            took_advice += 1
    criteria4[subject] = took_advice
print(f"finished criteria4")



# check criteria 5
# for each subject, check the longest reaction time
criteria5 = {}
for subject in criteria3:
    max_rt = float('-inf')
    for rt in criteria3[subject]['reaction_times']:
        # if rt is a tuple
        if isinstance(rt, tuple):
            if rt[0] > max_rt:
                max_rt = rt[0]
            if rt[1] > max_rt:
                max_rt = rt[1]
        else:
            if rt > max_rt:
                max_rt = rt
    criteria5[subject] = max_rt
print(f"finished criteria5")
       
# save all the criteria as pickle file
import pickle
with open('criteria.pkl', 'wb') as outfile:
    pickle.dump({'criteria1': criteria1, 'criteria2': criteria2, 'criteria3': criteria3, 'criteria4': criteria4, 'criteria5': criteria5}, outfile)