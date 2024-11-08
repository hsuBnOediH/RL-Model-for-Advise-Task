% Authors: Feng, Nov 2024 
% 
% Main Script for Model Free RL for advising Task
% 
dbstop if error
rng('default');
clear all;

% When runnng on cluster, read this var will be not empty
ON_CLUSTER = getenv('ON_CLUSTER');
% Detect the system
% 'pc' for Windows, 'mac' for local Mac, 'cluster' for running on VM cluster
env_sys = '';
if ispc
    env_sys = 'pc';
elseif ismac
    env_sys = 'mac';
elseif isunix
    env_sys = 'cluster';
else
    disp('Unknown operating system.');
end



% SWITCHES
% True -> Generate simulated behavior
SIM = false;
if ON_CLUSTER
    SIM = getenv('SIM');
end 
% True -> Fit the behavior data into the model
FIT = true;
if ON_CLUSTER
    FIT = getenv('FIT');
end 

% True -> TODO
PLOT = false;
if ON_CLUSTER
    PLOT = getenv('PLOT');
end 



% SETTINGS
% Subject identifier for the test or experiment, if on cluster read from ENV var
FIT_SUBJECT = 'FENGTEST';
if ON_CLUSTER
    FIT_SUBJECT = getenv('FIT_SUBJECT');
end

% ROOT:
% If ROOT is not assigned (i.e., empty), the script will derive the root 
% path based on the location of the main file.
ROOT = ''; 
if isempty(ROOT)
    ROOT = fileparts(mfilename('fullpath'));
    disp(['ROOT path set to: ', ROOT]);
end


% RES_PATH:
% If RES_PATH is not assigned (i.e., empty), it will be auto-generated relative to ROOT.
% If RES_PATH is a relative path, it will be appended to the ROOT path.
RES_PATH = '../../results/model_free/debug/';
if ON_CLUSTER
    RES_PATH = getenv('RES_PATH');
elseif ~isAbsolutePath(RES_PATH)
    RES_PATH = fullfile(ROOT, RES_PATH);
end

% INPUT_PATH:
% The folder path where the subject file is located. If INPUT_PATH is a relative path,
% it will be appended to the ROOT path.
INPUT_PATH = '../../inputs/';
if ON_CLUSTER
    INPUT_PATH = getenv('INPUT_PATH');
elseif ~isAbsolutePath(INPUT_PATH)
    INPUT_PATH = fullfile(ROOT, INPUT_PATH);
end

% IDX_CANDIDATE:
% This will define which candidate (set of parameters) is currently in use
% Modify this value to switch between different candidates (1 to 10 in this case)
IDX_CANDIDATE = 1; % Default to candidate 1, can be changed dynamically
if ON_CLUSTER
    env_value = getenv('IDX_CANDIDATE');
    IDX_CANDIDATE = str2double(env_value);
end


% Display all settings and switches
disp('--- Settings and Switches ---');
disp(['SIM (Simulate Behavior): ', num2str(SIM)]);
disp(['FIT (Fit Behavior Data): ', num2str(FIT)]);
disp(['PLOT (Plot Results): ', num2str(PLOT)]);
disp(['ON_CLUSTER (Example Subject): ', ON_CLUSTER]);
disp(['FIT_SUBJECT (Subject Identifier): ', FIT_SUBJECT]);
disp(['ROOT Path: ', ROOT]);
disp(['RES_PATH (Results Path): ', RES_PATH]);
disp(['INPUT_PATH (Input Path): ', INPUT_PATH]);
disp(['Environment System: ', env_sys]);
disp(['IDX_CANDIDATE: ', num2str(IDX_CANDIDATE)]);
disp('-----------------------------');

% Add external paths depending on the system
if strcmp(env_sys, 'pc')
    spmPath = 'L:/rsmith/all-studies/util/spm12/';
    spmDemPath = 'L:/rsmith/all-studies/util/spm12/toolbox/DEM/';
    tutorialPath = 'L:/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
   
elseif strcmp(env_sys, 'mac')
    spmPath =  [ROOT '/../../spm/'];
    spmDemPath = [ROOT '/../../spm/toolbox/DEM/'];
    tutorialPath = [ROOT '/../../Active-Inference-Tutorial-Scripts-main'];

elseif strcmp(env_sys, 'cluster')
    spmPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12';
    spmDemPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12/toolbox/DEM';
    tutorialPath = '/mnt/dell_storage/labs/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
end

addpath(spmPath);
addpath(spmDemPath);
addpath(tutorialPath);


all_params = struct(...
    'lr', 0.8, ...
    'inv_temp', 4, ...
    'discount_factor', 4, ...
    'l_loss_value', 1);


% Define an array of 10 field combinations (cell arrays)
all_fields = {
    {'lr', 'inv_temp', 'discount_factor', 'l_loss_value'}, ...
    {'lr', 'inv_temp', 'discount_factor', 'reward_value', 'l_loss_value'}, ...

};

all_fixeds = {
    {'omega', 0, 'eta', 1, 'state_exploration', 1, 'parameter_exploration', 0}, ...
    {'eta', 1, 'state_exploration', 1, 'parameter_exploration', 0}, ...
};


field_params = all_fields{IDX_CANDIDATE}; % Retrieve the field for the given candidate
fixed_params = all_fixeds{IDX_CANDIDATE}; % Retrieve the fixed parameters for the given candidate
params = struct();
fields = field_params;
for i = 1:length(field_params)
    params.(field_params{i}) =  all_params.(field_params{i});
end
for i = 1:2:length(fixed_params)
    params.(fixed_params{i}) = fixed_params{i+1};
end




% Check conditions and perform actions based on FIT and SIM settings
if FIT && ~SIM
    % If only fitting is required
    disp('Performing fitting only...');

    % data processing
    preprocessed_data = get_preprocessed_data(FIT_SUBJECT,INPUT_PATH);

    % inite the Q table, R table
    q_model = init_q_learning_model(params);

    % start update
    model = model_update(q_model,preprocessed_data);
    

elseif FIT && SIM
    disp('Performing fitting and simulation');

elseif ~FIT && SIM
    % If only simulation is required
    disp('Performing simulation only...');
    % Add your simulation code here
    % Example: simulate_model(candidate_params);
else
    % If neither fitting nor simulation is enabled
    disp('No fitting or simulation to perform.');
end


% Define the isAbsolutePath function
function isAbs = isAbsolutePath(givenPath)
    if ispc
        isAbs = length(givenPath) >= 2 && givenPath(2) == ':';
    elseif isunix || ismac
        isAbs = strncmp(givenPath, '/', 1);
    else
        error('Unknown operating system.');
    end
end

