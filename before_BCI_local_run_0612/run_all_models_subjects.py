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
#TODO  FORGETtoZero = false and True for all the six models

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
    }
}


run_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# Default MATLAB executable path (override with MATLAB_CMD env var if needed)
matlab_executable = os.environ.get('MATLAB_CMD', '/Applications/MATLAB_R2024a.app/bin/matlab')

preprocess_folder_path = "preprocessed/"
models_subfolders = [f.path for f in os.scandir(preprocess_folder_path) if f.is_dir()]

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



# loop thought model_params_dict to run the matlab script for each subject
for model, params in model_params_dict.items():
    print(f"\n=== Processing model: {model} ===")
    # TODO refactor param Name
    for pram_name, param_df in params.items():
        # Convert the DataFrame to a dictionary for MATLAB
        param_dict = param_df.to_dict(orient='list')
        print(f"\t Parameter: {pram_name}")
        # loop though each row of the DataFrame and convert it to a dictionary
        # TODO: only loop thought 1st row for now, later we can loop through all rows
        len_param_df = 1
        for i in range(len_param_df):
            row = param_df.iloc[i]
            # Convert the row to a dictionary
            row_dict = row.to_dict()
            # for each key in row_dict, remove the prefix 'fixed_' or 'posterior_'
            row_dict = {k.replace('fixed_', '').replace('posterior_', ''): v for k, v in row_dict.items()}

            # Set the environment variable for MATLAB
            # creat a empty environment dictionary
            env = os.environ.copy()
            env['SUBJECT_ID'] =  row_dict['id']
            env['RESULTS_DIR'] = f"result/{run_time}/{model}/{pram_name}/"
            # get the last digt of string param_name
            env['PARAMCOMBI'] = pram_name[-1]
            # Add model specific settings to the environment
            if model in model_settings:
                env['OMEGAdiff'] = str(model_settings[model]['OMEGAdiff'])
                env['OMEGAPOSINEGA'] = str(model_settings[model]['OMEGAPOSINEGA']).lower()
                env['FORGETtoZero'] = str(model_settings[model]['FORGETtoZero']).lower()
            else:
                print(f"[WARNING] Model {model} not found in model_settings, using default settings.")
                raise ValueError(f"Model {model} not found in model_settings. Please check the model name.")

            # Set the environment variables for each parameter
            for key, value in row_dict.items():
                if key in var_to_env:
                    env[var_to_env[key]] = str(value)
                else:
                    if key in ['id', 'lamgda']:
                        continue
                    print(f"[WARNING] {key} not found in var_to_env, skipping.")
                    # warning
                    raise ValueError(f"Variable {key} not found in var_to_env mapping. Please check the variable names.")

            # use the env to run the MATLAB command
            print(f"\n=== Processing {row_dict['id']} with model {model} and parameters {pram_name} ===")
            print(f"Environment variables: {env}")
            code, out, err = run_matlab_for_subject(row_dict['id'], env)
            if code != 0:
                print(f"[ERROR] MATLAB exited with code {code} for {row_dict['id']}")
                print("--- STDOUT ---")
                print(out.strip())
                print("--- STDERR ---")
                print(err.strip())
            else:
                print(f"[OK] Finished {row_dict['id']} with model {model} and parameters {pram_name}")
                # Optionally, print out for debugging:
                # print(out.strip())

