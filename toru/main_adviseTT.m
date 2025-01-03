% main script for fitting/simming behavior on the advise task
clear variables;
dbstop if error
rng('default');
cd(fileparts(mfilename('fullpath')));

SIM = true; % Generate simulated behavior (if false and FIT == true, will fit to subject file data instead)
FIT = true; % Fit example subject data 'BBBBB' or fit simulated behavior (if SIM == true)
plot = true;
%indicate if prolific or local
local = false;

% Setup directories based on system
if ispc
    root = 'L:';
    results_dir = 'L:/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/ATresults';
    FIT_SUBJECT = '6544b95b7a6b86a8cd8feb88' ; %  6550ea5723a7adbcc422790b 5afa19a4f856320001cf920f(No advice participant) TORUTEST
    %INPUT_DIRECTORY = [root '/rsmith/wellbeing/tasks/AdviceTask/behavioral_files_2-6-24'];  % Where the subject file is located
    INPUT_DIRECTORY = [root '/NPC/DataSink/StimTool_Online/WB_Advice'];  % Where the subject file is located
    INPUT_DIRECTORYforSIM = [root '/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/resultsforallmodels/model_5'];  % Where the subject file is located

else
    root = '/media/labs';
    FIT_SUBJECT = getenv('SUBJECT');
    results_dir = getenv('RESULTS');
    if ~SIM
      INPUT_DIRECTORY = getenv('INPUT_DIRECTORY');  % Where the subject file is located
    elseif SIM
      INPUT_DIRECTORYforSIM = getenv('INPUT_DIRECTORYforSIM');
    end

end


if SIM
    fprintf([INPUT_DIRECTORYforSIM '\n']);
end

if FIT && ~SIM
    fprintf([INPUT_DIRECTORY '\n']);
end

fprintf([FIT_SUBJECT '\n']);



addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);
addpath([root '/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main']);
addpath([root '/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask']);


% Define all parameters passed into the model; specify which ones to fit in
% field

field = {'p_a','inv_temp','reward_value','l_loss_value','omega','eta_d_win','eta_d_loss','eta_a'}; %those are fitted

model = 1; %Specify model 1 = active inference, 2 = RL connected, 3 = RL disconnected

% fit reward value and loss value, fix explore weight to 1, fix novelty
% weight to 0

if SIM

    directorysim = dir(INPUT_DIRECTORYforSIM);
    index_array = find(arrayfun(@(n) contains(directorysim(n).name, ['advise_task-' FIT_SUBJECT]),1:numel(directorysim)));
    filesim = [INPUT_DIRECTORYforSIM '/' directorysim(index_array).name];
    subdatsim = readtable(filesim);

paramsim.p_a = subdatsim.posterior_p_a;
paramsim.inv_temp = subdatsim.posterior_inv_temp;
paramsim.reward_value = subdatsim.posterior_reward_value;
paramsim.l_loss_value = subdatsim.posterior_l_loss_value;
paramsim.omega = subdatsim.posterior_omega;
%paramsim.omega_d = subdatsim.posterior_omega_d;
%paramsim.omega_d_win = subdatsim.posterior_omega_d_win;
%paramsim.omega_d_loss = subdatsim.posterior_omega_d_loss;
%paramsim.omega_a = subdatsim.posterior_omega_a;
%paramsim.omega_a_win = subdatsim.posterior_omega_a_win;
%paramsim.omega_a_loss = subdatsim.posterior_omega_a_loss;
%paramsim.eta = subdatsim.posterior_eta;
%paramsim.eta_d = subdatsim.eta_d;
paramsim.eta_d_win = subdatsim.posterior_eta_d_win;
paramsim.eta_d_loss = subdatsim.posterior_eta_d_loss;
paramsim.eta_a = subdatsim.posterior_eta_a;
%paramsim.eta_a_win = subdatsim.posterior_eta_a_win;
%paramsim.eta_a_loss = subdatsim.posterior_eta_a_loss;
%paramsim.lamgda = subdatsim.posterior_lamgda;
paramsim.state_exploration = 1;
paramsim.parameter_exploration = 0;

    [gen_data] = advise_simTT(paramsim, plot, model);
end
    
if FIT

params.p_a = .8;
params.inv_temp = 1;
params.reward_value = 4; %for Active inference
%params.reward_value = 1; %for RL
params.l_loss_value = 4; %for Active inference
%params.l_loss_value = 8; %for RL
%params.omega = 0; %As fixed param
params.omega = .2; %As prior
%params.omega_d = .2;
%params.omega_d_win = .2;
%params.omega_d_loss = .2;
%params.omega_a = .2;
%params.omega_a_win = .2;
%params.omega_a_loss = .2;
%params.eta = 1; %As fixed param
%params.eta = .5; %As prior
%params.eta_d = .5;
params.eta_d_win = .5;
params.eta_d_loss = .5;
params.eta_a = .5;
%params.eta_a_win = .5;
%params.eta_a_loss = .5;
%params.lamgda = .5;
params.state_exploration = 1;
params.parameter_exploration = 0;

    if SIM
        [fit_results, DCM] = advise_sim_fitTT(FIT_SUBJECT, INPUT_DIRECTORYforSIM, gen_data, field, params, plot, model);
    else
    
        if ~local
            [fit_results, DCM] = Advice_fit_prolificTT(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot, model);
        else
            [fit_results, DCM] = Advice_fit(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot, model);
        end
        
        model_free_results = advise_mf_TT(fit_results.file);
        
        
        mf_fields = fieldnames(model_free_results);
        for i=1:length(mf_fields)
            fit_results.(mf_fields{i}) = model_free_results.(mf_fields{i});      
        end
        
    end

    
end
 
 fit_results.F = DCM.F;
 fit_results.modelAIorRL = model;
      
 currentDateTimeString = datestr(now, 'yyyy-mm-dd_HH-MM-SS');  

 writetable(struct2table(fit_results), [results_dir '/advise_task-' FIT_SUBJECT '_' currentDateTimeString '_fits.csv']); 
 saveas(gcf,[results_dir '/' FIT_SUBJECT '_' currentDateTimeString '_fit_plot.png']);
 save(fullfile([results_dir '/fit_results_' FIT_SUBJECT '_' currentDateTimeString '.mat']), 'DCM');
                            
