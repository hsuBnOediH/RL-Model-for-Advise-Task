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

% IS_CONNECTED:
% This will define if the left option change will connectely change the right option 
IS_CONNECTED = true;
if ON_CLUSTER
    IS_CONNECTED = getenv('IS_CONNECTED');
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
disp(['IS_CONNECTED: ', num2str(IS_CONNECTED)]);
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


% Define the parameters for the model, both fixed and free
% left_better: if left is better than right, according to p_right in AInf Code
% advise_truthness: the probability of the advice being truthful, according to p_a in AInf Code
% inv_temp: inverse temperature
% outcome_sensitivity: sensitivity to outcome, used while initializing the Q table
% large_loss_sensitive: sensitivity to large loss(large party block), used while initializing the Q table

% forgetting_rate: forgetting rate for all unchosen actions
% learning_rate: learning rate for all chosen actions

% with_advice_learning_rate: learning rate for all chosen actions with advice, has to be used with without_advice_learning_rate
% without_advice_learning_rate: learning rate for all chosen actions without advice, has to be used with with_advice_learning_rate

% with_adive_win_learning_rate: learning rate for win trials with advice, has to be used with with_advice_loss_learning_rate, without_advice_win_learning_rate, without_advice_loss_learning_rate
% with_advice_loss_learning_rate: learning rate for lose trials with advice, has to be used with with_adive_win_learning_rate, without_advice_win_learning_rate, without_advice_loss_learning_rate
% without_advice_win_learning_rate: learning rate for win trials without advice, has to be used with with_adive_win_learning_rate, with_advice_loss_learning_rate, without_advice_loss_learning_rate
% without_advice_loss_learning_rate: learning rate for lose trials without advice, has to be used with with_adive_win_learning_rate, with_advice_loss_learning_rate, without_advice_win_learning_rate

% with_advise_forgetting_rate: forgetting rate for all unchosen actions with advice, has to be used with without_advise_forgetting_rate
% without_advise_forgetting_rate: forgetting rate for all unchosen actions without advice, has to be used with with_advise_forgetting_rate

% with_advice_win_forgetting_rate: forgetting rate for win trials with advice, has to be used with with_advice_loss_forgetting_rate, without_advice_win_forgetting_rate, without_advice_loss_forgetting_rate
% with_advice_loss_forgetting_rate: forgetting rate for lose trials with advice, has to be used with with_advice_win_forgetting_rate, without_advice_win_forgetting_rate, without_advice_loss_forgetting_rate
% without_advice_win_forgetting_rate: forgetting rate for win trials without advice, has to be used with with_advice_win_forgetting_rate, with_advice_loss_forgetting_rate, without_advice_loss_forgetting_rate
% without_advice_loss_forgetting_rate: forgetting rate for lose trials without advice, has to be used with with_advice_win_forgetting_rate, with_advice_loss_forgetting_rate, without_advice_win_forgetting_rate

% discount_factor: discount factor for future rewards, fixed to 1

all_params = struct(...
    'left_better', 0.5, ...
    'advise_truthness', 0.8, ...
    'inv_temp', 4, ...
    'outcome_sensitivity',4, ...
    'large_loss_sensitive', 8, ...
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
    'without_advice_loss_forgetting_rate', 0.2, ...
    'discount_factor', 1 ...
    );

% all the parameters need to be transformed into [0,1]
zero_one_fields = {'left_better','advise_truthness','learning_rate','with_advice_learning_rate','without_advice_learning_rate','with_advice_win_learning_rate','with_advice_loss_learning_rate',...
    'without_advice_win_learning_rate','without_advice_loss_learning_rate','forgetting_rate','with_advice_forgetting_rate','without_advice_forgetting_rate',...
    'with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate','discount_factor'};
% all the parameters need to be transformed into positive values, logit transformation
% todo: the range of discount factor is [0,1] or [0,inf]?
positive_fields = {'inv_temp','outcome_sensitivity',...
    };

% for 10 candidates, we have 10 different fields settings, _fields is for free parameters, fix_fields is for fixed parameters
fit_fields = {
    {'inv_temp','outcome_sensitivity','large_loss_sensitive'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','forgetting_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','forgetting_rate','learning_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','forgetting_rate','with_advice_learning_rate','without_advice_learning_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','forgetting_rate','with_advice_learning_rate','without_advice_win_learning_rate','without_advice_loss_learning_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','forgetting_rate','with_advice_win_learning_rate','with_advice_loss_learning_rate','without_advice_learning_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','with_advice_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_forgetting_rate'}, ...
    {'inv_temp','outcome_sensitivity','large_loss_sensitive','with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate','learning_rate'}, ...
};
fix_fields ={
    {'left_better','advise_truthness','learning_rate','forgetting_rate','discount_factor'}, ...
    {'left_better','advise_truthness','learning_rate','discount_factor'}, ...
    {'left_better','advise_truthness','discount_factor'}, ...
    {'left_better','advise_truthness','discount_factor'}, ...
    {'left_better','advise_truthness','discount_factor'}, ...
    {'left_better','advise_truthness','discount_factor'}, ...
    {'left_better','advise_truthness','learning_rate','discount_factor'}, ...
    {'left_better','advise_truthness','learning_rate','discount_factor'}, ...
    {'left_better','advise_truthness','learning_rate','discount_factor'}, ...
    {'left_better','advise_truthness','discount_factor'}, ...
};


% assemble the parameters for the current candidate based on the IDX_CANDIDATE
field_params = fit_fields{IDX_CANDIDATE}; 
params = struct();
fields = field_params;
for i = 1:length(field_params)
    params.(field_params{i}) =  all_params.(field_params{i});
end

fixed_params = struct();
for i = 1:length(fix_fields{IDX_CANDIDATE})
    fixed_params.(fix_fields{IDX_CANDIDATE}{i}) =  all_params.(fix_fields{IDX_CANDIDATE}{i});
end

% Check conditions and perform actions based on FIT and SIM settings
if FIT && ~SIM
    % data processing from the raw subject csv file
    preprocessed_data = get_preprocessed_data(FIT_SUBJECT,INPUT_PATH);
    % if there is no valid data for this subject, end the script
    if isempty(preprocessed_data)
        disp('No data to fit');
        return;
    end


    % inite the model
    q_model = init_q_learning_model(params,all_params);

    % overwrite the log likelihood function function
    M.L = @(P,M,U,Y)log_likelihood_func(P, M, U, Y);
    % assign the free parameters with convar maxtrix
    M.pC= q_model.con_var;
    M.pE = q_model.params; 
    M.fixed_params = fixed_params;
    M.q_table = q_model.q_table;
    M.is_connected = IS_CONNECTED;
    % Y wont be used in the log likelihood function, so just assign it as the same as U
    U = preprocessed_data;
    Y = preprocessed_data;
    % call the spm_nlsi_Newton function to fit the model
    [Ep, Cp, F] = spm_nlsi_Newton(M, U, Y);

    % transfor eP into valid range
    fields = fieldnames(Ep);
    for i = 1:length(fields)
         field = fields{i};
         if ismember(field, zero_one_fields)
            Ep.(field) = 1/(1+exp(-Ep.(field)));
        elseif ismember(field, positive_fields)
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

