% Authors: Feng, Jan 2024
% 
% Main Script for Model Free RL for advising Task
% 
dbstop if error
rng('default');
clear all;
clear variables;
plot = true;

ON_CLUSTER = getenv('ON_CLUSTER');

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
    % tanform the char to logical
    SIM = strcmp(SIM, 'True');
end
% True -> Fit the behavior data into the model
FIT = true;
if ON_CLUSTER
    FIT = getenv('FIT');
    % tanform the char to logical
    FIT = strcmp(FIT, 'True');
end 

PLOT = true;
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
RES_PATH = '../../outputs/model_free/debug/';
if ON_CLUSTER
    RES_PATH = getenv('RES_PATH');
end

% INPUT_PATH:
% The folder path where the subject file is located. If INPUT_PATH is a relative path,
% it will be appended to the ROOT path.
INPUT_PATH = '../../inputs/';
if ON_CLUSTER
    INPUT_PATH = getenv('INPUT_PATH');
end

% MODEL_IDX:
% Specify model 1 = active inference, 2 = RL connected, 3 = RL disconnected, 4 = active inference with message passing
MODEL_IDX = 4; % Default to candidate 1, can be changed dynamically
if ON_CLUSTER
    env_value = getenv('MODEL_IDX');
    MODEL_IDX= str2double(env_value);
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
disp(['MODEL_IDX: ', num2str(MODEL_IDX)]);
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

    INPUT_DIRECTORY = [ROOT '/' INPUT_PATH];

elseif strcmp(env_sys, 'cluster')
    spmPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12';
    spmDemPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12/toolbox/DEM';
    tutorialPath = '/mnt/dell_storage/labs/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
    addpath('/media/labs/rsmith/lab-members/fli/advise_task/RL-Model-for-Advise-Task/feng/model_free/')
    INPUT_DIRECTORY = [ROOT '/' INPUT_PATH];
end

addpath(spmPath);
addpath(spmDemPath);
addpath(tutorialPath);


%%% Specify 
IFLAMGDA = false;
ONEMODEL = false;


% fit reward value and loss value, fix explore weight to 1, fix novelty
% weight to 0


for paramcombi = 1:5
    if SIM
        directorysim = dir(INPUT_DIRECTORYforSIM);
        index_array = find(arrayfun(@(n) contains(directorysim(n).name, ['advise_task-' FIT_SUBJECT]),1:numel(directorysim)));
        % Check if FIT_SUBJECT exists
        if isempty(index_array)
            fprintf('FIT_SUBJECT "%s" not found in %s. Ending process.\n', FIT_SUBJECT, INPUT_DIRECTORYforSIM);
            return; % End the process
        end
        filesim = [INPUT_DIRECTORYforSIM '/' directorysim(index_array).name];
        subdatsim = readtable(filesim);
        paramsim.state_exploration = 1;
        paramsim.parameter_exploration = 0;
        paramsim.p_a = .8;
        paramsim.inv_temp = 1;
        paramsim.reward_value = 1;  % 4 in the original model
        paramsim.l_loss_value = 1;  % 4 in the original model
        paramsim.omega = .2;
        paramsim.eta_d_win = .5;
        paramsim.eta_d_loss = .5;
        paramsim.eta_a = .5;
        paramsim.Rsensitivity = 0;

        [gen_data] = advise_simTT(paramsim, plot, MODEL_IDX);  

    end

    if FIT

        clear params

        params.p_a = .8;
        params.inv_temp = 1;
        params.state_exploration = 1;
        params.parameter_exploration = 0;
        params.Rsensitivity = 2; 
    
        if MODEL_IDX == 1 %for Active inference
            params.reward_value = 1; % 4 in the original model
            params.l_loss_value = 1; % 4 in the original model
        elseif MODEL_IDX == 4 %for Active inference with policy
            params.reward_value = 1; % 4 in the original model
            params.l_loss_value = 1; % 4 in the original model
        elseif MODEL_IDX ~= 1 % for RL
            params.reward_value = 1; 
            params.l_loss_value = 4; % 8 in the original model
        end
    
        if ONEMODEL 
            params.omega = .2;
            params.eta_d_win = .5;
            params.eta_d_loss = .5;
            params.eta_a = .5;
            params.lamgda = 1; %As fixed param
            field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity','state_exploration'};
        else
            if paramcombi == 1
                % TODO: remove when runing on cluster
                is_debugging = false;

                if is_debugging
                    params.omega = .6;
                    params.eta = .6;
                    params.p_a = .75;
                    params.inv_temp = 2;
                else
                    params.omega = .2;
                    params.eta = .5;
        
                end
                if MODEL_IDX == 1
                    field = {'p_a','inv_temp','omega','eta','state_exploration', 'Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 4
                    field = {'p_a','inv_temp','omega','eta','state_exploration', 'Rsensitivity'}; %those are fitted
                elseif MODEL_IDX ~= 1 && MODEL_IDX ~= 4
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega','eta','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega','eta','Rsensitivity'};
                    end
                end
    
            elseif paramcombi == 2
                params.omega = .2;
                params.eta_d = .5;
                params.eta_a = .5;
                if MODEL_IDX == 1
                    field = {'p_a','inv_temp','omega','eta_d','eta_a', 'state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 4
                    field = {'p_a','inv_temp','omega','eta_d','eta_a', 'state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX ~= 1 && MODEL_IDX ~= 4
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a','Rsensitivity'};
                    end
                end
            elseif paramcombi == 3
                params.omega = .2;
                params.eta_d_win = .5;
                params.eta_d_loss = .5;
                params.eta_a = .5;
                if MODEL_IDX == 1
                    field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 4
                    field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX ~= 1 && MODEL_IDX ~= 4
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d_win','eta_d_loss','eta_a','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                    end
                end
            elseif paramcombi == 4
                params.omega = .2;
                params.eta_d = .5;
                params.eta_a_win = .5;
                params.eta_a_loss = .5;      
                if MODEL_IDX == 1
                    field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 4
                    field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX ~= 1 && MODEL_IDX ~= 4
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                    end
                end
            elseif paramcombi == 5
                params.omega_d_win = .2;
                params.omega_d_loss = .2;
                params.omega_a_win = .2;
                params.omega_a_loss = .2;
                params.eta = .5;
                if MODEL_IDX == 1 
                    field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 4
                    field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
                elseif MODEL_IDX == 2
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                    end
                elseif MODEL_IDX == 3
                    if IFLAMGDA
                        params.lamgda = .5;
                        field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                    else
                        params.lamgda = 1; %As fixed param
                        field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                    end
                end
            end
            
        end

        if SIM
            [fit_results, DCM] = advise_sim_fit_uni(FIT_SUBJECT, INPUT_DIRECTORYforSIM, gen_data, field, params, plot, MODEL_IDX);
        else
            [fit_results, DCM] = advice_fit_prolific_uni(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot, MODEL_IDX);
            model_free_results = advise_mf_uni(fit_results.file);
            mf_fields = fieldnames(model_free_results);
            for i=1:length(mf_fields)
                fit_results.(mf_fields{i}) = model_free_results.(mf_fields{i});      
            end
        end

        fit_results.F = DCM.F;
        fit_results.modelAIorRL = MODEL_IDX;
        if ~ONEMODEL
            results_dir = fullfile(ROOT, RES_PATH);
            % Define the folder name dynamically based on paramcombi
            folder_name = fullfile(results_dir, ['paramcombi' num2str(paramcombi)]);
            % Check if the folder exists; if not, create it
            if ~exist(folder_name, 'dir')
                mkdir(folder_name);
            end
            % Save the table to the folder
            writetable(struct2table(fit_results), ...
                fullfile(folder_name, ['advise_task-' FIT_SUBJECT  '_fits.csv']));
            % Save the plot to the folder
            saveas(gcf, fullfile(folder_name, [FIT_SUBJECT  '_fit_plot.png']));
            % Save the .mat file to the folder
            save(fullfile(folder_name, ['fit_results_' FIT_SUBJECT  '.mat']), 'DCM');
        else
            writetable(struct2table(fit_results), [results_dir '/advise_task-' FIT_SUBJECT '_fits.csv']); 
            saveas(gcf,[results_dir '/' FIT_SUBJECT '_fit_plot.png']);
            save(fullfile([results_dir '/fit_results_' FIT_SUBJECT '.mat']), 'DCM');
        end
    end
end
    
    
    
    
    
    