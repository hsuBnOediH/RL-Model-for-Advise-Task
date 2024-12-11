% main script for fitting/simming behavior on the advise task
clear variables;
dbstop if error
rng('default');
cd(fileparts(mfilename('fullpath')));

SIM = false; % Generate simulated behavior (if false and FIT == true, will fit to subject file data instead)
FIT = true; % Fit example subject data 'BBBBB' or fit simulated behavior (if SIM == true)
plot = true;
%indicate if prolific or local
local = false;

% Setup directories based on system
if ispc
    root = 'L:';
    results_dir = 'L:/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/ATresults';
    FIT_SUBJECT = '6544b95b7a6b86a8cd8feb88'; % 6544b95b7a6b86a8cd8feb88 6550ea5723a7adbcc422790b
    %INPUT_DIRECTORY = [root '/rsmith/wellbeing/tasks/AdviceTask/behavioral_files_2-6-24'];  % Where the subject file is located
    INPUT_DIRECTORY = [root '/NPC/DataSink/StimTool_Online/WB_Advice'];  % Where the subject file is located

else
    root = '/media/labs';
    FIT_SUBJECT = getenv('SUBJECT');
    results_dir = getenv('RESULTS');
    INPUT_DIRECTORY = getenv('INPUT_DIRECTORY');

end


fprintf([INPUT_DIRECTORY '\n']);
fprintf([FIT_SUBJECT '\n']);



addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);
addpath([root '/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main']);
addpath([root '/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask']);

% Define all parameters passed into the model; specify which ones to fit in
% field
params.p_a = .75;
params.inv_temp = 4;
params.reward_value = 4;
%params.l_loss_value = 4; %for Active inference
params.l_loss_value = 8; %for RL
%params.omega = .2; %As prior
params.omega = 0;
% params.omega_d_win = .2;
% params.omega_d_loss = .2;
% params.omega_a_win = .2;
% params.omega_a_loss = .2;
%params.omega_d = .2;
%params.omega_a = .2;
%params.eta = .5; %As prior
params.eta = 1;
%params.eta_d = .5;
%params.eta_d_win = .5;
%params.eta_d_loss = .5;
%params.eta_a = .5;
%params.eta_a_win = .5;
%params.eta_a_loss = .5;
params.state_exploration = 1;
params.parameter_exploration = 0;
params.lamgda = 0.5;

field = {'p_a','inv_temp','reward_value','l_loss_value','lamgda'}; %those are fitted


% fit reward value and loss value, fix explore weight to 1, fix novelty
% weight to 0

if SIM
    [gen_data] = advise_simTT(params, plot);
end
    
if FIT
    if SIM
        fit_results = advise_sim_fitTT(gen_data, field, priors);
    else
    
        if ~local
            [fit_results, DCM] = Advice_fit_prolificTT(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot);
        else
            [fit_results, DCM] = Advice_fit(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot);
        end
        
        model_free_results = advise_mf(fit_results.file);
        
        
        mf_fields = fieldnames(model_free_results);
        for i=1:length(mf_fields)
            fit_results.(mf_fields{i}) = model_free_results.(mf_fields{i});      
        end
        currentDateTimeString = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        writetable(struct2table(fit_results), [results_dir '/advise_task-' FIT_SUBJECT '_' currentDateTimeString '_fits.csv']);
    end

    
end

    
    

 saveas(gcf,[results_dir '/' FIT_SUBJECT '_' currentDateTimeString '_fit_plot.png']);
 save(fullfile([results_dir '/fit_results_' FIT_SUBJECT '_' currentDateTimeString '.mat']), 'DCM');
                            
