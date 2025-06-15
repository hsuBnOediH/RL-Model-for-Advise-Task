#!/usr/bin/env python3
"""
batch_run_matlab.py

Loop over a list of subject IDs and invoke MATLAB for each one in batch mode.
Assumes you have a MATLAB function `process_subject(subject_id)` on your MATLAB path.
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_matlab_for_subject(subject_id,env):

    # Set SUBJECT_ID environment variable for MATLAB and ensure script directory is cwd
    script_dir = Path(__file__).parent.resolve()

    # Build the MATLAB command: call process_subject and then exit
    matlab_cmd = [
        matlab_executable,
        '-nodisplay',
        '-nosplash',
        # '-desktop',
        '-r',
        # switch to script directory, call main_adviseTT which reads getenv('SUBJECT_ID')
        f"try, cd('{script_dir}'), main_adviseTT, catch ME, disp(getReport(ME)), exit(1), end, exit(0);"
        # switch to script directory, set breakpoint at entry, then run
        # f"cd('{script_dir}'); dbstop in main_adviseTT at 1; main_adviseTT"
    ]

    # Run it
    proc = subprocess.run(matlab_cmd,
                          capture_output=True,
                          text=True,
                          env=env,
                          cwd=str(script_dir))
    return proc.returncode, proc.stdout, proc.stderr

def process_run(row_dict, model, pram_name):
    # assemble environment
    env = os.environ.copy()
    env['SUBJECT_ID'] = row_dict['id']
    env['RESULTS_DIR'] = f"result/{run_time}/{model}/{pram_name}/"
    env['PARAMCOMBI'] = pram_name[-1]
    if model in model_settings:
        env['OMEGAdiff'] = str(model_settings[model]['OMEGAdiff'])
        env['OMEGAPOSINEGA'] = str(model_settings[model]['OMEGAPOSINEGA']).lower()
        env['FORGETtoZero'] = str(model_settings[model]['FORGETtoZero']).lower()


    # set parameter envs
    for key, value in row_dict.items():
        if key in var_to_env:
            env[var_to_env[key]] = str(value)
    # run
    start_time = time.time()
    code, out, err = run_matlab_for_subject(row_dict['id'], env)
    pbar.update(1)
    os.makedirs(env['RESULTS_DIR'], exist_ok=True)
    # write logs
    with open(os.path.join(env['RESULTS_DIR'], f"{row_dict['id']}_out.log"), 'w') as f_out:
        f_out.write(out)
    with open(os.path.join(env['RESULTS_DIR'], f"{row_dict['id']}_err.log"), 'w') as f_err:
        f_err.write(err)
    duration = time.time() - start_time
    with open(os.path.join(env['RESULTS_DIR'], f"{row_dict['id']}_time.log"), 'w') as f_time:
        f_time.write(f"{duration}\n")
    print(f"\t{model}/{pram_name}/{row_dict['id']} took {duration:.2f}s")

# map from stripped variable name → ENV_VAR name
var_to_env = {
    'p_a'                    : 'P_A',
    'inv_temp'               : 'INV_TEMP',
    'state_exploration'      : 'STATE_EXPLORATION',
    'parameter_exploration'  : 'PARAMETER_EXPLORATION',
    'Rsensitivity'           : 'RSENSITIVITY',
    'reward_value'           : 'REWARD_VALUE',
    'l_loss_value'           : 'L_LOSS_VALUE',
    'eta'                    : 'ETA',
    'eta_d'                  : 'ETA_D',
    'eta_a'                  : 'ETA_A',
    'eta_d_win'              : 'ETA_D_WIN',
    'eta_d_loss'             : 'ETA_D_LOSS',
    'eta_a_win'              : 'ETA_A_WIN',
    'eta_a_loss'             : 'ETA_A_LOSS',
    'omega'                  : 'OMEGA',
    'omega_d'                : 'OMEGA_D',
    'omega_a'                : 'OMEGA_A',
    'omega_a_posi'           : 'OMEGA_A_POSI',
    'omega_a_nega'           : 'OMEGA_A_NEGA',
    'omega_d_posi'           : 'OMEGA_D_POSI',
    'omega_d_nega'           : 'OMEGA_D_NEGA',
}

model_settings = {
    "outputMerged_RLdisconnectedwolamgdarsallfreeRadconomegaFRtozero": {
        "OMEGAdiff": 2,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": True
    },
    "outputMerged_RLdisconnectedwolamgdarsallfreeRadPNconomega": {
        "OMEGAdiff": 3,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": False
    },
    "outputMerged_RLdisconnectedwolamgdarsallfreeRoneomegaFRtozero": {
        "OMEGAdiff": 1,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": True
    },
    "outputMerged_RLdisconnectedwolamgdarsallfreeRadconomega": {
        "OMEGAdiff": 2,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": False
    },
    "outputMerged_RLdisconnectedwolamgdarsallfreeRoneomega": {
        "OMEGAdiff": 1,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": False
    },
    "outputMerged_RLdisconnectedwolamgdarsallfreeRadPNconomegaFRtozero": {
        "OMEGAdiff": 3,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": True
    },
    "outputMerged_ActiveinferenceSecondisRforgetadviceUnchosen":{
        "OMEGAdiff": 1,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": False
    },
    "outputMerged_ActiveinferenceSecondisRforgetadviceUnchosenFRtozero": {
        "OMEGAdiff": 1,
        "OMEGAPOSINEGA": True,
        "FORGETtoZero": True
    }
}


run_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# Default MATLAB executable path (override with MATLAB_CMD env var if needed)
matlab_executable = os.environ.get('MATLAB_CMD', '/Applications/MATLAB_R2024a.app/bin/matlab')

preprocess_folder_path = "preprocessed/"
models_subfolders = [f.path for f in os.scandir(preprocess_folder_path) if f.is_dir()]
# run for Active inference models only
models_subfolders = [f for f in models_subfolders if f.split('/')[-1].startswith("outputMerged_Act")]

model_params_dict = {}
for subfolder in models_subfolders:
    subfolder_last = subfolder.split('/')[-1]
    files = [f.path for f in os.scandir(subfolder) if f.is_file()]
    for file in files:
        # read the dataframe put into the dictionary
        df = pd.read_csv(file)
        if subfolder_last not in model_params_dict:
            model_params_dict[subfolder_last] = {}
        file_name_last = file.split('_')[-1]
        # remove the .csv extension from the file name
        file_name_last = file_name_last.split('.')[0]

        model_params_dict[subfolder_last][file_name_last] = df

# compute total number of MATLAB runs
total_runs = sum(len(df) for params in model_params_dict.values() for df in params.values())
pbar = tqdm(total=total_runs, desc="Running MATLAB subjects")
# record overall start time
overall_start = time.time()



for model, params in model_params_dict.items():
    # collect all tasks for this model, subproecess code
    tasks = []
    with ThreadPoolExecutor(max_workers=min(sum(len(df) for df in params.values()), os.cpu_count() or 1)) as executor:
        for pram_name, param_df in params.items():
            for _, row in param_df.iterrows():
                row_dict = {k.replace('fixed_', '').replace('posterior_', ''): v for k, v in row.to_dict().items()}
                tasks.append(executor.submit(process_run, row_dict, model, pram_name))
        # wait for completion
        for _ in as_completed(tasks):
            pass


# close progress bar after all models processed
pbar.close()

# print total runtime
overall_end = time.time()
total_duration = overall_end - overall_start
print(f"Total runtime: {total_duration:.2f} seconds")
