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
    'inv_temp', 4, ...
    'reward_sensitivity', 4, ...
    'loss_sensitivity', 4, ...
    'forgetting_rate', 0.2, ...
    'learning_rate', 0.5, ...
    'with_advice_learning_rate', 0.5, ...
    'without_advice_learning_rate', 0.5, ...
    'with_adive_win_learning_rate', 0.5, ...
    'with_advice_loss_learning_rate', 0.5, ...
    'without_advice_win_learning_rate', 0.5, ...
    'without_advice_loss_learning_rate', 0.5, ...
    'with_advise_forgetting_rate', 0.2, ...
    'without_advise_forgetting_rate', 0.2, ...
    'with_advice_win_forgetting_rate', 0.2, ...
    'with_advice_loss_forgetting_rate', 0.2, ...
    'without_advice_win_forgetting_rate', 0.2, ...
    'without_advice_loss_forgetting_rate', 0.2,...
    'discount_factor', 1 ...
    );

% Define an array of 10 field combinations (cell arrays)
all_fields = {
    {'inv_temp','reward_sensitivity','loss_sensitivity'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','forgetting_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','forgetting_rate','learning_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','forgetting_rate','with_advice_learning_rate','without_advice_learning_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','forgetting_rate','with_advice_learning_rate','without_advice_win_learning_rate','without_advice_loss_learning_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','forgetting_rate','with_advice_win_learning_rate','with_advice_loss_learning_rate','without_advice_learning_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','with_advice_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_forgetting_rate'}, ...
    {'inv_temp','reward_sensitivity','loss_sensitivity','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate','learning_rate'}, ...
};
fixed_fields ={
    {'learning_rate','forgetting_rate','discount_factor'}, ...
    {'learning_rate','discount_factor'}, ...
    {'discount_factor'}, ...
    {'discount_factor'}, ...
    {'discount_factor'}, ...
    {'discount_factor'}, ...
    {'learning_rate','discount_factor'}, ...
    {'learning_rate','discount_factor'}, ...
    {'learning_rate','discount_factor'}, ...
    {'discount_factor'}, ...
};

field_params = all_fields{IDX_CANDIDATE}; % Retrieve the field for the given candidate
params = struct();
fields = field_params;
for i = 1:length(field_params)
    params.(field_params{i}) =  all_params.(field_params{i});
end

fixed_params = struct();
for i = 1:length(fixed_fields{IDX_CANDIDATE})
    fixed_params.(fixed_fields{IDX_CANDIDATE}{i}) =  all_params.(fixed_fields{IDX_CANDIDATE}{i});
end




% Check conditions and perform actions based on FIT and SIM settings
if FIT && ~SIM
    % If only fitting is required
    disp('Performing fitting only...');

    % data processing
    preprocessed_data = get_preprocessed_data(FIT_SUBJECT,INPUT_PATH);

    % inite the Q table, R table
    mb_model = init_model_based_model(params);



    % start update
    % LL = fitting_model(q_model,preprocessed_data);

    M.L = @(P,M,U,Y)mb_log_likelihood_func(P, M, U, Y);
    % variance all 0.5 for each fitted parameter, nxn sparse matrix
    M.pC= mb_model.con_var;
    M.pE = mb_model.params; 
    M.fixed_params = fixed_params;
    M.prob_table = mb_model.prob_table;
    % M.trialinfo = q_model.trialinfo;
    U = preprocessed_data;
    Y = preprocessed_data;
    [Ep, Cp, F] = spm_nlsi_Newton(M, U, Y);

    % transfor eP into valid range
    fields = fieldnames(Ep);
    for i = 1:length(fields)
         % for lr and discount_factor range 0-1
         field = fields{i};

         if ismember(field, {'lr','lr_advice','lr_self','lr_left','lr_right','lr_win','lr_loss','discount_factor'})
            Ep.(field) = 1/(1+exp(-Ep.(field)));
        
        elseif ismember(field, {'inv_temp'})
            Ep.(field) = log(1+exp(Ep.(field)));
        end
    end

    
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

