% main script for fitting/simming behavior on the advise taskclear variables;dbstop if error
rng('default');
cd(fileparts(mfilename('fullpath')));

SIM = false; % Generate simulated behavior (if false and FIT == true, will fit to subject file data instead)
FIT = true; % Fit example subject data 'BBBBB' or fit simulated behavior (if SIM == true)
plot = true;
%indicate if prolific or local
local = false;



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

% Setup directories based on system
if ispc
    ROOT = 'L:';
    results_dir = 'L:/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/ATresults';
    FIT_SUBJECT = '6544b95b7a6b86a8cd8feb88'; %  6550ea5723a7adbcc422790b 5afa19a4f856320001cf920f(No advice participant)  TORUTEST
    %INPUT_DIRECTORY = [root '/rsmith/wellbeing/tasks/AdviceTask/behavioral_files_2-6-24'];  % Where the subject file is located
    INPUT_DIRECTORY = [root '/NPC/DataSink/StimTool_Online/WB_Advice'];  % Where the subject file is located
    INPUT_DIRECTORYforSIM = [root '/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/resultsforallmodels/RLdisconnectedwolamgdarsNoforgetall/paramcombi4'];  % Where the subject file is located
elseif ismac
    ROOT = '';
    results_dir = getenv('RESULTS_DIR');
    if isempty(results_dir)
        results_dir = '/Users/fengli/Documents/RL-Model-for-Advise-Task/before_BCI_local_run_0612/result/debug';
    end
    % read subject from environment variable
    FIT_SUBJECT = getenv('SUBJECT_ID');
    if isempty(FIT_SUBJECT)
        FIT_SUBJECT = '53b98f20fdf99b472f4700e4';
    end
    INPUT_DIRECTORY = '/Users/fengli/Documents/RL-Model-for-Advise-Task/DataSink';  % Where the subject file is located
    INPUT_DIRECTORYforSIM = [ROOT '/rsmith/lab-members/ttakahashi/WellbeingTasks/AdviceTask/resultsforallmodels/RLdisconnectedwolamgdarsNoforgetall/paramcombi4'];
else
    ROOT = '/media/labs';
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


if strcmp(env_sys, 'pc')
    spmPath = 'L:/rsmith/all-studies/util/spm12/';
    spmDemPath = 'L:/rsmith/all-studies/util/spm12/toolbox/DEM/';
    tutorialPath = 'L:/rsmith/lab-members/cgoldman/Active-Inference-Tutorial-Scripts-main';
   
elseif strcmp(env_sys, 'mac')
    spmPath = '/Users/fengli/Documents/spm/';
    spmDemPath = '/Users/fengli/Documents/spm/toolbox/DEM/';
    tutorialPath = '/Users/fengli/Documents/RL-Model-for-Advise-Task/Active-Inference-Tutorial-Scripts-main';

    INPUT_DIRECTORY = [ROOT '/' INPUT_DIRECTORY];

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

%%% Specify model 1 = active inference, 2 = RL connected (outdated), 3 = RL disconnected
model = 1; 

%%% Specify 
IFLAMGDA = false;
ONEMODEL = false;

OMEGAPOSINEGA = getenv('OMEGAPOSINEGA');
if isempty(OMEGAPOSINEGA)
    OMEGAPOSINEGA = false; % default to false if not set
end
if ischar(OMEGAPOSINEGA)
    OMEGAPOSINEGA = strcmpi(OMEGAPOSINEGA, 'true'); % Convert string to boolean
end

OMEGAdiff = getenv('OMEGAdiff');
if isempty(OMEGAdiff)
    OMEGAdiff = 1; % default to 1 if not set
end
if ischar(OMEGAdiff)
    OMEGAdiff = str2double(OMEGAdiff); % Convert string to number
end
%OMEGAdiff = 4; % 1 = oneomega, 2 = omega for context and ad, 3 = omega for context and ad (posi vs. nega), 4 = omega for context (posi vs. nega) and ad (posi vs. nega)

% fit reward value and loss value, fix explore weight to 1, fix novelty
% weight to 0


%for paramcombi = 1:4 %for connected, or posi nega forgetting version

for paramcombi = 1
   paramcombi = getenv('PARAMCOMBI');
   if isempty(paramcombi)
      paramcombi = 1; % default to 1 if not set
   else
      paramcombi = str2double(paramcombi); % Convert string to number
   end

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

      % %paramsim.state_exploration = subdatsim.posterior_state_exploration;
      % paramsim.parameter_exploration = 0;
      % 
      % paramsim.p_a = subdatsim.posterior_p_a;
      % paramsim.inv_temp = subdatsim.posterior_inv_temp;
      % %paramsim.reward_value = subdatsim.posterior_reward_value;
      % paramsim.reward_value = 1;  % 4 in the original model
      % paramsim.l_loss_value = subdatsim.posterior_l_loss_value;
      % %paramsim.l_loss_value = 1;  % 4 in the original model
      % 
      % %paramsim.omega = subdatsim.posterior_omega;
      % %paramsim.omega_d = subdatsim.posterior_omega_d;
      % %paramsim.omega_d_win = subdatsim.posterior_omega_d_win;
      % %paramsim.omega_d_loss = subdatsim.posterior_omega_d_loss;
      % %paramsim.omega_a = subdatsim.posterior_omega_a;
      % %paramsim.omega_a_win = subdatsim.posterior_omega_a_win;
      % %paramsim.omega_a_loss = subdatsim.posterior_omega_a_loss;
      % %paramsim.eta = subdatsim.posterior_eta;
      % paramsim.eta_d = subdatsim.posterior_eta_d;
      % %paramsim.eta_d_win = subdatsim.posterior_eta_d_win;
      % %paramsim.eta_d_loss = subdatsim.posterior_eta_d_loss;
      % %paramsim.eta_a = subdatsim.posterior_eta_a;
      % paramsim.eta_a_win = subdatsim.posterior_eta_a_win;
      % paramsim.eta_a_loss = subdatsim.posterior_eta_a_loss;
      % %paramsim.lamgda = subdatsim.posterior_lamgda;
      % paramsim.lamgda = 1;
      % paramsim.Rsensitivity = subdatsim.posterior_Rsensitivity;

      %%%to play around
      paramsim.state_exploration = 1;
      paramsim.parameter_exploration = 0;

      paramsim.p_a = .5;
      paramsim.inv_temp = 2;
      %paramsim.reward_value = subdatsim.posterior_reward_value;
      paramsim.reward_value = 4;  % 4 in the original model
      %paramsim.l_loss_value = subdatsim.posterior_l_loss_value;
      paramsim.l_loss_value = 4;  % 4 in the original model

      paramsim.omega = .1;
      %paramsim.omega_d = subdatsim.posterior_omega_d;
      %paramsim.omega_d_win = subdatsim.posterior_omega_d_win;
      %paramsim.omega_d_loss = subdatsim.posterior_omega_d_loss;
      %paramsim.omega_a = subdatsim.posterior_omega_a;
      %paramsim.omega_a_win = subdatsim.posterior_omega_a_win;
      %paramsim.omega_a_loss = subdatsim.posterior_omega_a_loss;
      %paramsim.eta = subdatsim.posterior_eta;
      paramsim.eta_d = .3;
      %paramsim.eta_d_win = .5;
      %paramsim.eta_d_loss = .5;
      %paramsim.eta_a = .5;
      paramsim.eta_a_win = .3;
      paramsim.eta_a_loss = .3;
      %paramsim.lamgda = subdatsim.posterior_lamgda;
      paramsim.lamgda = 1;
      paramsim.Rsensitivity = 2;

      [gen_data] = advise_simTT(paramsim, plot, model);

   end
    
   if FIT
      clear params

      params.p_a = getenv('P_A');
      if isempty(params.p_a)
         params.p_a = .8; % default value if not set
      else
         params.p_a = str2double(params.p_a); % Convert string to number
      end
      
      
      params.inv_temp = getenv('INV_TEMP');
      if isempty(params.inv_temp)
         params.inv_temp = 1; % default value if not set
      else
         params.inv_temp = str2double(params.inv_temp); % Convert string to number
      end

      params.state_exploration = getenv('STATE_EXPLORATION');
      if isempty(params.state_exploration)
         params.state_exploration = 1; % default value if not set
      else
         params.state_exploration = str2double(params.state_exploration); % Convert string to number
      end

      params.parameter_exploration = getenv('PARAMETER_EXPLORATION');
      if isempty(params.parameter_exploration)
         params.parameter_exploration = 0; % default value if not set
      else
         params.parameter_exploration = str2double(params.parameter_exploration); % Convert string to number
      end


      params.Rsensitivity = getenv('RSENSITIVITY');
      if isempty(params.Rsensitivity)
         params.Rsensitivity = 2; % default value if not set
      else
         params.Rsensitivity = str2double(params.Rsensitivity); % Convert string to number
      end

      if model == 1 %for Active inference
         params.reward_value = 1; % 4 in the original model
         params.l_loss_value = 1; % 4 in the original model
      elseif model ~= 1 % for RL
         params.reward_value = getenv('REWARD_VALUE');
         if isempty(params.reward_value)
            params.reward_value = 4; % default value if not set
         else
            params.reward_value = str2double(params.reward_value); % Convert string to number
         end
         params.l_loss_value = getenv('L_LOSS_VALUE');
         if isempty(params.l_loss_value)
            params.l_loss_value = 4; % default value if not set
         else
            params.l_loss_value = str2double(params.l_loss_value); % Convert string to number
         end
      end

      if ONEMODEL  
         %tentative for one model checking
         params.eta = .5;
         params.lamgda = 1; %As fixed param
         if OMEGAdiff == 1
         
            params.omega = .2;
      

            if model == 1
               field = {'state_exploration', 'p_a','inv_temp','omega','eta','Rsensitivity'};
            else
               field = {'p_a','inv_temp','l_loss_value','omega','eta','Rsensitivity'};
            end
      
         elseif OMEGAdiff == 2

            params.omega_d = .2;
            params.omega_a = .2;
            
            if model == 1

               field = {'state_exploration', 'p_a','inv_temp','omega_d','omega_a','eta','Rsensitivity'};
               %field = {'p_a','inv_temp','omega','eta','Rsensitivity','reward_value'};
            else 
               field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta','Rsensitivity'};
            end

         elseif OMEGAdiff == 3
         end

      else

         if paramcombi == 1
            params.eta = getenv('ETA');
            if isempty(params.eta)
               params.eta = .5; % default value if not set
            else
               params.eta = str2double(params.eta); % Convert string to number
            end

            if OMEGAdiff == 1
      
               params.omega = getenv('OMEGA');
               if isempty(params.omega)
                  params.omega = .2; % default value if not set
               else
                  params.omega = str2double(params.omega); % Convert string to number
               end
      
               if model == 1
                  field = {'p_a','inv_temp','omega','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'reward_value','inv_temp','p_a','omegaposi','omeganega','eta','Rsensitivity'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega','eta','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 2

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end

               params.omega_a = getenv('OMEGA_A');
               if isempty(params.omega_a)
                  params.omega_a = .2; % default value if not set
               else
                  params.omega_a = str2double(params.omega_a); % Convert string to number
               end
               
               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'reward_value','inv_temp','p_a','omega','eta','Rsensitivity'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 3

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end
               
               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a_posi','omega_a_nega','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'reward_value','inv_temp','p_a','omega','eta','Rsensitivity'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 4

               params.omega_d_posi = getenv('OMEGA_D_POSI');
               if isempty(params.omega_d_posi)
                  params.omega_d_posi = .2; % default value if not set
               else
                  params.omega_d_posi = str2double(params.omega_d_posi); % Convert string to number
               end
            
               params.omega_d_nega = getenv('OMEGA_D_NEGA');
               if isempty(params.omega_d_nega)
                  params.omega_d_nega = .2; % default value if not set
               else
                  params.omega_d_nega = str2double(params.omega_d_nega); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end
            
               if model == 1
                  field = {'p_a','inv_temp','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'reward_value','inv_temp','p_a','omega','eta','Rsensitivity'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta','Rsensitivity'};
                  end
               end

            end

         elseif paramcombi == 2
   
            params.eta_d = getenv('ETA_D');
            if isempty(params.eta_d)
               params.eta_d = .5; % default value if not set
            else
               params.eta_d = str2double(params.eta_d); % Convert string to number
            end
            params.eta_a = getenv('ETA_A');
            if isempty(params.eta_a)
               params.eta_a = .5; % default value if not set
            else
               params.eta_a = str2double(params.eta_a); % Convert string to number
            end

            if OMEGAdiff == 1
               
               params.omega = getenv('OMEGA');
               if isempty(params.omega)
                  params.omega = .2; % default value if not set
               else
                  params.omega = str2double(params.omega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega','eta_d','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omegaposi','omeganega','eta_d','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega','eta_d','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a','Rsensitivity'};
                  end
               end


            elseif OMEGAdiff == 2

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a = getenv('OMEGA_A');
               if isempty(params.omega_a)
                  params.omega_a = .2; % default value if not set
               else
                  params.omega_a = str2double(params.omega_a); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a','eta_d','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta_d','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta_d','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 3

            params.omega_d = getenv('OMEGA_D');
            if isempty(params.omega_d)
               params.omega_d = .2; % default value if not set
            else
               params.omega_d = str2double(params.omega_d); % Convert string to number
            end
            params.omega_a_posi = getenv('OMEGA_A_POSI');
            if isempty(params.omega_a_posi)
               params.omega_a_posi = .2; % default value if not set
            else
               params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
            end
            params.omega_a_nega = getenv('OMEGA_A_NEGA');
            if isempty(params.omega_a_nega)
               params.omega_a_nega = .2; % default value if not set
            else
               params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
            end

            if model == 1
               field = {'p_a','inv_temp','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a','state_exploration','Rsensitivity'}; %those are fitted
               %field = {'p_a','inv_temp','omega','eta_d','eta_a','Rsensitivity','reward_value'};
            elseif model ~= 1
               if IFLAMGDA
                  params.lamgda = .5;
                  field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a','lamgda','Rsensitivity'}; %those are fitted
               else
                  params.lamgda = 1; %As fixed param
                  field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a','Rsensitivity'};
                  %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a','Rsensitivity'};
               end
            end

            elseif OMEGAdiff == 4

               params.omega_d_posi = getenv('OMEGA_D_POSI');
               if isempty(params.omega_d_posi)
                  params.omega_d_posi = .2; % default value if not set
               else
                  params.omega_d_posi = str2double(params.omega_d_posi); % Convert string to number
               end
               params.omega_d_nega = getenv('OMEGA_D_NEGA');
               if isempty(params.omega_d_nega)
                  params.omega_d_nega = .2; % default value if not set
               else
                  params.omega_d_nega = str2double(params.omega_d_nega); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
            
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a','Rsensitivity'};
                  end
               end

            end


         elseif paramcombi == 3

            params.eta_d_win = getenv('ETA_D_WIN');
            if isempty(params.eta_d_win)
               params.eta_d_win = .5; % default value if not set
            else
               params.eta_d_win = str2double(params.eta_d_win); % Convert string to number
            end
            params.eta_d_loss = getenv('ETA_D_LOSS');
            if isempty(params.eta_d_loss)
               params.eta_d_loss = .5; % default value if not set
            else
               params.eta_d_loss = str2double(params.eta_d_loss); % Convert string to number
            end
            params.eta_a = getenv('ETA_A');
            if isempty(params.eta_a)
               params.eta_a = .5; % default value if not set
            else
               params.eta_a = str2double(params.eta_a); % Convert string to number
            end

            if OMEGAdiff == 1
               
               params.omega = getenv('OMEGA');
               if isempty(params.omega)
                  params.omega = .2; % default value if not set
               else
                  params.omega = str2double(params.omega); % Convert string to number
               end

            if model == 1
               field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
               %field = {'p_a','inv_temp','omegaposi','omeganega','eta_d_win','eta_d_loss','eta_a','Rsensitivity','reward_value'};
            elseif model ~= 1
               if IFLAMGDA
                  params.lamgda = .5;
                  field = {'p_a','inv_temp','l_loss_value','omega','eta_d_win','eta_d_loss','eta_a','lamgda','Rsensitivity'}; %those are fitted
               else
                  params.lamgda = 1; %As fixed param
                  field = {'p_a','inv_temp','reward_value','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                  %field = {'p_a','inv_temp','l_loss_value','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
               end
            end

            elseif OMEGAdiff == 2

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a = getenv('OMEGA_A');
               if isempty(params.omega_a)
                  params.omega_a = .2; % default value if not set
               else
                  params.omega_a = str2double(params.omega_a); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta_d_win','eta_d_loss','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 3

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 4

               params.omega_d_posi = getenv('OMEGA_D_POSI');
               if isempty(params.omega_d_posi)
                  params.omega_d_posi = .2; % default value if not set
               else
                  params.omega_d_posi = str2double(params.omega_d_posi); % Convert string to number
               end
               params.omega_d_nega = getenv('OMEGA_D_NEGA');
               if isempty(params.omega_d_nega)
                  params.omega_d_nega = .2; % default value if not set
               else
                  params.omega_d_nega = str2double(params.omega_d_nega); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d_win','eta_d_loss','eta_a','Rsensitivity','reward_value'};
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d_win','eta_d_loss','eta_a','Rsensitivity'};
                  end
               end

            end


         elseif paramcombi == 4
            params.eta_d = getenv('ETA_D');
            if isempty(params.eta_d)
               params.eta_d = .5; % default value if not set
            else
               params.eta_d = str2double(params.eta_d); % Convert string to number
            end
            params.eta_a_win = getenv('ETA_A_WIN');
            if isempty(params.eta_a_win)
               params.eta_a_win = .5; % default value if not set
            else
               params.eta_a_win = str2double(params.eta_a_win); % Convert string to number
            end
            params.eta_a_loss = getenv('ETA_A_LOSS');
            if isempty(params.eta_a_loss)
               params.eta_a_loss = .5; % default value if not set
            else
               params.eta_a_loss = str2double(params.eta_a_loss); % Convert string to number
            end
     

            if OMEGAdiff == 1
               
               params.omega = getenv('OMEGA');
               if isempty(params.omega)
                  params.omega = .2; % default value if not set
               else
                  params.omega = str2double(params.omega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omegaposi','omeganega','eta_d','eta_a_win','eta_a_loss','Rsensitivity','reward_value'}; 
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega','eta_d','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 2

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a = getenv('OMEGA_A');
               if isempty(params.omega_a)
                  params.omega_a = .2; % default value if not set
               else
                  params.omega_a = str2double(params.omega_a); % Convert string to number
               end
               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','Rsensitivity','reward_value'}; 
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta_d','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 3

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','Rsensitivity','reward_value'}; 
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 4

               params.omega_d_posi = getenv('OMEGA_D_POSI');
               if isempty(params.omega_d_posi)
                  params.omega_d_posi = .2; % default value if not set
               else
                  params.omega_d_posi = str2double(params.omega_d_posi); % Convert string to number
               end
               params.omega_d_nega = getenv('OMEGA_D_NEGA');
               if isempty(params.omega_d_nega)
                  params.omega_d_nega = .2; % default value if not set
               else
                  params.omega_d_nega = str2double(params.omega_d_nega); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  field = {'p_a','inv_temp','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega','eta_d','eta_a_win','eta_a_loss','Rsensitivity','reward_value'}; 
               elseif model ~= 1
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                     %field = {'p_a','inv_temp','l_loss_value','eta_d','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            end

         elseif paramcombi == 5
      
            % params.omega_d_win = .2;
            % params.omega_d_loss = .2;
            % params.omega_a_win = .2;
            % params.omega_a_loss = .2;
            % params.eta = .5;

            params.eta_d_win = getenv('ETA_D_WIN');
            if isempty(params.eta_d_win)
               params.eta_d_win = .5; % default value if not set
            else
               params.eta_d_win = str2double(params.eta_d_win); % Convert string to number
            end
            params.eta_d_loss = getenv('ETA_D_LOSS');
            if isempty(params.eta_d_loss)
               params.eta_d_loss = .5; % default value if not set
            else
               params.eta_d_loss = str2double(params.eta_d_loss); % Convert string to number
            end
            params.eta_a_win = getenv('ETA_A_WIN');
            if isempty(params.eta_a_win)
               params.eta_a_win = .5; % default value if not set
            else
               params.eta_a_win = str2double(params.eta_a_win); % Convert string to number
            end
            params.eta_a_loss = getenv('ETA_A_LOSS');
            if isempty(params.eta_a_loss)
               params.eta_a_loss = .5; % default value if not set
            else
               params.eta_a_loss = str2double(params.eta_a_loss); % Convert string to number
            end

            if OMEGAdiff == 1
      
               params.omega = getenv('OMEGA');
               if isempty(params.omega)
                  params.omega = .2; % default value if not set
               else
                  params.omega = str2double(params.omega); % Convert string to number
               end

               if model == 1
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity','reward_value'};
                  field = {'p_a','inv_temp','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','omega','state_exploration','Rsensitivity'};
               elseif model == 2
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega','eta','Rsensitivity'};
                  end
               elseif model == 3
                  if IFLAMGDA
                     params.lamgda = .5;
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                     field = {'p_a','inv_temp','l_loss_value','omega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; 
                  else
                     params.lamgda = 1; %As fixed param
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                     field = {'p_a','inv_temp','reward_value','omega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 2

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a = getenv('OMEGA_A');
               if isempty(params.omega_a)
                  params.omega_a = .2; % default value if not set
               else
                  params.omega_a = str2double(params.omega_a); % Convert string to number
               end

               if model == 1
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity','reward_value'};
                  field = {'p_a','inv_temp','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','omega_d','omega_a','state_exploration','Rsensitivity'};
               elseif model == 2
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta','Rsensitivity'};
                  end
               elseif model == 3
                  if IFLAMGDA
                     params.lamgda = .5;
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; 
                  else
                     params.lamgda = 1; %As fixed param
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 3

               params.omega_d = getenv('OMEGA_D');
               if isempty(params.omega_d)
                  params.omega_d = .2; % default value if not set
               else
                  params.omega_d = str2double(params.omega_d); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end

               if model == 1
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
                  %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity','reward_value'};
                  field = {'p_a','inv_temp','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','omega_d','omega_a_posi','omega_a_nega','state_exploration','Rsensitivity'};
               elseif model == 2
                  if IFLAMGDA
                     params.lamgda = .5;
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta','lamgda','Rsensitivity'}; %those are fitted
                  else
                     params.lamgda = 1; %As fixed param
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta','Rsensitivity'};
                  end
               elseif model == 3
                  if IFLAMGDA
                     params.lamgda = .5;
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                     field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; 
                  else
                     params.lamgda = 1; %As fixed param
                     %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                     field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','Rsensitivity'};
                  end
               end

            elseif OMEGAdiff == 4

               params.omega_d_posi = getenv('OMEGA_D_POSI');
               if isempty(params.omega_d_posi)
                  params.omega_d_posi = .2; % default value if not set
               else
                  params.omega_d_posi = str2double(params.omega_d_posi); % Convert string to number
               end
               params.omega_d_nega = getenv('OMEGA_D_NEGA');
               if isempty(params.omega_d_nega)
                  params.omega_d_nega = .2; % default value if not set
               else
                  params.omega_d_nega = str2double(params.omega_d_nega); % Convert string to number
               end
               params.omega_a_posi = getenv('OMEGA_A_POSI');
               if isempty(params.omega_a_posi)
                  params.omega_a_posi = .2; % default value if not set
               else
                  params.omega_a_posi = str2double(params.omega_a_posi); % Convert string to number
               end
               params.omega_a_nega = getenv('OMEGA_A_NEGA');
               if isempty(params.omega_a_nega)
                  params.omega_a_nega = .2; % default value if not set
               else
                  params.omega_a_nega = str2double(params.omega_a_nega); % Convert string to number
               end


            if model == 1
               %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','state_exploration','Rsensitivity'}; %those are fitted
               %field = {'p_a','inv_temp','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity','reward_value'};
               field = {'p_a','inv_temp','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','state_exploration','Rsensitivity'};
            elseif model == 2
               if IFLAMGDA
                  params.lamgda = .5;
                  field = {'p_a','inv_temp','l_loss_value','omega_d','omega_a_posi','omega_a_nega','eta','lamgda','Rsensitivity'}; %those are fitted
               else
                  params.lamgda = 1; %As fixed param
                  field = {'p_a','inv_temp','reward_value','omega_d','omega_a_posi','omega_a_nega','eta','Rsensitivity'};
               end
            elseif model == 3
               if IFLAMGDA
                  params.lamgda = .5;
                  %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','lamgda','Rsensitivity'}; %those are fitted
                  field = {'p_a','inv_temp','l_loss_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','lamgda','Rsensitivity'}; 
               else
                  params.lamgda = 1; %As fixed param
                  %field = {'p_a','inv_temp','l_loss_value','omega_d_win','omega_d_loss','omega_a_win','omega_a_loss','eta','Rsensitivity'};
                  field = {'p_a','inv_temp','reward_value','omega_d_posi','omega_d_nega','omega_a_posi','omega_a_nega','eta_d_win','eta_d_loss','eta_a_win','eta_a_loss','Rsensitivity'};
               end
            end

         end

      end

   end

   if SIM
         [fit_results, DCM] = advise_sim_fitTT(FIT_SUBJECT, INPUT_DIRECTORYforSIM, gen_data, field, params, plot, model);
   else
      
      if ~local
            [fit_results, DCM] = Advice_fit_prolificTT(FIT_SUBJECT, INPUT_DIRECTORY, params, field, plot, model, OMEGAPOSINEGA);
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

   % There is No F since no F been compute
   % fit_results.F = DCM.F;
   fit_results.modelAIorRL = model;
      
   currentDateTimeString = datestr(now, 'yyyy-mm-dd_HH-MM-SS');  

   modelStr = num2str(model);

   if ~ONEMODEL

      % Define the folder name dynamically based on paramcombi
      folder_name = results_dir;

      % Check if the folder exists; if not, create it
      if ~exist(folder_name, 'dir')
         mkdir(folder_name);
      end

      % Save the table to the folder
      writetable(struct2table(fit_results), ...
         fullfile(folder_name, ['advise_task-' FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '_fits.csv']));

      % Save the plot to the folder
      saveas(gcf, fullfile(folder_name, [FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '_fit_plot.png']));

      % Save the .mat file to the folder
      save(fullfile(folder_name, ['fit_results_' FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '.mat']), 'DCM');

   else

      writetable(struct2table(fit_results), [results_dir '/advise_task-' FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '_fits.csv']); 
      saveas(gcf,[results_dir '/' FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '_fit_plot.png']);
      save(fullfile([results_dir '/fit_results_' FIT_SUBJECT '_AIorRL' modelStr '_' currentDateTimeString '.mat']), 'DCM');

   end

end
%end





