import pandas as pd
import os
import datetime
from tqdm import tqdm

subjects_folder_path = "../inputs/raw"
# subjects_folder_path = "/mnt/dell_storage/labs/rsmith/wellbeing/data/raw"
subjects_id_file_path =os.path.join(subjects_folder_path,  "coop_local_ids.csv")


result_folder_path = f"../outputs/local/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# result_folder_path = f"/mnt/dell_storage/labs/rsmith/lab-members/fli/advise_task/local_data/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
# create the result folder if not exists
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)

# read the ids from the csv file
ids = []
with open(subjects_id_file_path, "r") as f:
    ids_lines = f.readlines()
    # skip the first line
    for line in ids_lines[1:]:
        line = line.strip()
        if line:
            ids.append(line)


# the subjects folder path find according subject folder the format is f"sub-{id}"
for id in tqdm(ids, desc="Processing subjects"):
    print(f"Processing subject {id}...")
    subject_folder_path = os.path.join(subjects_folder_path, f"sub-{id}")
    if not os.path.exists(subject_folder_path):
        print(f"Subject folder {subject_folder_path} does not exist.")
        continue
    # find the subject folder and list all files in the folder
    subject_task_files = os.listdir(subject_folder_path)
    # find the file format is f"{id}-T0-__AT_R1-_BEH.csv"
    # if not exist, print the message
    task_file = None
    for file in subject_task_files:
        if file.startswith(f"{id}-T0-") and file.endswith("AT_R1-_BEH.csv"):
            task_file = file
            break
    if not task_file:
        print(f"Task file for subject {id} does not exist.")
        continue
    # read the task file
    task_file_path = os.path.join(subject_folder_path, task_file)
    task_df = pd.read_csv(task_file_path)
    # remove the header row and use the first row as the header
    task_df.columns = task_df.iloc[0]
    task_df = task_df[1:]



    # need to process the task_df to get the result_df, each row is a trial, and the columns are the features
    # trial_idx, trial_info_1, trial_info_2, trial_info_3, final_choice, reward, advice, rt_1, rt_2
    result_dicts = {}
    for trial_idx in range(1, 361):
        result_dicts[trial_idx] = {
            "trial_info_1": None,
            "trial_info_2": None,
            "trial_info_3": None,
            "final_choice": None,
            "reward": None,
            "advice": None,
            "rt_1": None,
            "rt_2": None
        }

    # extract the data from the task_df that "event_code" is "4" and "trial_number" is trial_idx +1
    trial_info_rows = task_df[task_df["event_code"] == "4"]

    # for idx in range(360):
    #     # read each row "trial_type" and "response"
    #     trial_type = trial_info_rows.iloc[idx]["trial_type"]
    #     response = trial_info_rows.iloc[idx]["response"]
    #
    #     probs = response.split("_")
    #     trial_info_1 = float(probs[-1])
    #     trial_info_2 = float(probs[0])
    #     trial_info_3 = 40 if trial_type.startswith("sm") else 80
    #
    #     result_dicts[idx + 1]["trial_info_1"] = trial_info_1
    #     result_dicts[idx + 1]["trial_info_2"] = trial_info_2
    #     result_dicts[idx + 1]["trial_info_3"] = trial_info_3

    # extract the final choice and reward from the task_df where "event_code" is "8"
    final_choice_rows = task_df[task_df["event_code"] == "8"]
    for idx in range(360):
        final_choice = final_choice_rows.iloc[idx]["response"]
        reward = final_choice_rows.iloc[idx]["result"]
        result_dicts[idx + 1]["final_choice"] = final_choice
        result_dicts[idx + 1]["reward"] = float(reward)

    advice_onset_rows = task_df[task_df["event_code"] == "6"]
    advice_rows_idx = 0
    if advice_onset_rows.empty:
        print(f"Subject {id} has no advice onset rows!!! Consider removing this subject???")
        # if there is no advice onset, set all advice to None
        for idx in range(360):
            result_dicts[idx + 1]["advice"] = None
        # continue to the next subject
        continue
    else:
        for idx in range(360):
            trial_number = advice_onset_rows.iloc[advice_rows_idx]["trial_number"]
            trial_number = int(trial_number)  # Convert to integer
            if trial_number != idx:
                # didn't take advice for this trial,
                continue
            advice = advice_onset_rows.iloc[advice_rows_idx]["response"]
            result_dicts[idx + 1]["advice"] = advice
            advice_rows_idx += 1
            if advice_rows_idx >= len(advice_onset_rows):
                break



    for idx in range(360):
        # extract "trial_number" == "idx"
        trial_rows = task_df[task_df["trial_number"] == str(idx)]
        # if event_code == "6" exists, extreact absolute_time of "6" "5" "8" "9"
        # rt_1 is the absolute_time of "6" - "5", rt_2 is the absolute_time of "8" - "9"
        # if "6" does not exist, set rt_1 to None, rt2 is "8" - "5"
        rt_1 = None
        rt_2 = None
        if not trial_rows.empty:
            if "6" in trial_rows["event_code"].values:
                rt_1 = float(trial_rows[trial_rows["event_code"] == "6"]["absolute_time"].values[0]) - \
                       float(trial_rows[trial_rows["event_code"] == "5"]["absolute_time"].values[0])
                rt_2 = float(trial_rows[trial_rows["event_code"] == "8"]["absolute_time"].values[0]) - \
                   float(trial_rows[trial_rows["event_code"] == "9"]["absolute_time"].values[0])
            else:
                rt_2 = float(trial_rows[trial_rows["event_code"] == "8"]["absolute_time"].values[0]) - \
                       float(trial_rows[trial_rows["event_code"] == "5"]["absolute_time"].values[0])
        result_dicts[idx + 1]["rt_1"] = rt_1
        result_dicts[idx + 1]["rt_2"] = rt_2

    # convert the result_dicts to a DataFrame, the key should be transfer to trial_idx
    result_df = pd.DataFrame.from_dict(result_dicts, orient='index')
    result_df.reset_index(inplace=True)
    result_df.rename(columns={"index": "trial_idx"}, inplace=True)

    # save the result_df to a csv file
    result_file_path = os.path.join(result_folder_path, f"{id}.csv")
    result_df.to_csv(result_file_path, index=False)
















