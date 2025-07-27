function [fit_results, DCM] = Advice_fit_prolificTT(subject,folder,params,field, plot, model, OMEGAPOSINEGA, MODELBASED)
% initialize has_practice_effects to false, tracking if this participant's
% first complete behavioral file came after they played the task a little
% bit
has_practice_effects = false;
% Manipulate Data
directory = dir(folder);

% sort by date
dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
% Sort the dates and get the sorted indices
[~, sortedIndices] = sort(dates);
% Use the sorted indices to sort the structure array
sortedDirectory = directory(sortedIndices);

index_array = find(arrayfun(@(n) contains(sortedDirectory(n).name, ['active_trust_' subject]),1:numel(sortedDirectory)));
if length(index_array) > 1
    disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
end
file = '';
for k = 1:length(index_array)
    file_index = index_array(k);
    file = [folder '/' sortedDirectory(file_index).name];

    subdat = readtable(file);
    if strcmp(class(subdat.trial),'cell')
        subdat.trial = str2double(subdat.trial);
    end
    % make sure this file has correct number of trials
    if any(cellfun(@(x) isequal(x, 'MAIN'), subdat.trial_type)) && (max(subdat.trial) ~= 359)
        has_practice_effects = true;
    end
    if max(subdat.trial) ~= 359
        continue;
    end
  
    subdat = subdat(max(find(ismember(subdat.trial_type,'MAIN')))+1:end,:);


    compressed = subdat(subdat.event_type==4,:);

    % Prepare a new cell array to hold the split trial_type values
    trialinfo = cell(size(compressed,1), 3);
    % Loop over each unique trial to split the trial_type string
    for i = 1:size(compressed,1)
        parts = strsplit(compressed.trial_type{i}, '_');
        % Reassign the split parts to the new order and convert to numeric
        trialinfo{i, 2} = (parts{1}); % Second column
        trialinfo{i, 1} = (parts{3}); % First column
        trialinfo{i, 3} = (parts{4}); % Third column
    end

    % lets look at options selected
    response = subdat(subdat.event_type==8, :);
    % if the person managed to cause a glitch and select two bandits in one 
    % trial, use only the first one as response/result
    [~, idx] = unique(response.trial, 'first');
    response = response(idx, :);
    resp = response.response;
    result = subdat(subdat.event_type==9 & ~(strcmp(subdat.result,"try left")|strcmp(subdat.result,"try right")), :);
    points = result.result;
    % re = tp(~ismember(tp.result, {'try right', 'try left'}),:).result;
    % w_ad = tp(ismember(tp.result, {'try right', 'try left'}),{'trial' 'result'});

    got_advice = subdat.event_type ==9 & (strcmp(subdat.result,"try left")|strcmp(subdat.result,"try right"));
    trials_got_advice = subdat.trial(got_advice);
    advice_given = subdat.result(got_advice);
    trials_got_advice = trials_got_advice + 1;


    for n = 1:size(resp,1)
        % indicate if participant chose right or left
        if ismember(resp(n),'right')
            r=4;
        elseif ismember(resp(n),'left')
            r=3;
         elseif ismember(resp(n),'none')
            error("this person chose the did nothing option and our scripts are not set up to allow that")
        end 

        if str2double(points{n}) >0 
            pt=3;
        elseif str2double(points{n}) <0 
            pt=2;
        else
            error("this person chose the did nothing option and our scripts are not set up to allow that")
        end

        if ismember(n, trials_got_advice)
            u{n} = [1 2; 1 r]';
            index = find(trials_got_advice == n);
            if strcmp(advice_given{index}, 'try right')
                y = 3;
            elseif strcmp(advice_given{index}, 'try left')
                y = 2;
            end
            o{n} = [1 y 1; 1 1 pt; 1 2 r];
        else
            u{n} = [1 r; 1 1]';
            o{n} = [1 1 1; 1 pt 1; 1 r 1];
        end

        % get reaction time
        trial = n-1;
        trial_data = subdat(subdat.trial==trial,:);
        asked_advice = any(trial_data.event_type == 6);
        if ~asked_advice
            reaction_times{n}=[nan,trial_data.absolute_time(trial_data.event_type==8) - trial_data.absolute_time(trial_data.event_type==5)];
        else
            index = find(trial_data.event_type == 5,1);
            first_stim = trial_data.absolute_time(index);
            index = find(trial_data.event_type == 6,1);
            first_action = trial_data.absolute_time(index);
            index = find(trial_data.event_type == 5);
            index = index(2);
            second_stim = trial_data.absolute_time(index);
            index = find(trial_data.event_type == 8,1);
            second_action = trial_data.absolute_time(index);        

            reaction_times{n}=[first_action - first_stim,second_action - second_stim];
        end
    end
    trialinfo = trialinfo(:,:);

    %plotting old model
%     if plot
%             MDP     = advise_gen_model(trialinfo(:,:),params);
%             for idx_trial = 1:360
%                 MDP(idx_trial).o = o{idx_trial};
%                 MDP(idx_trial).u = u{idx_trial};
%                 MDP(idx_trial).reaction_times = reaction_times{idx_trial};
%             end
% 
%             MDP  = spm_MDP_VB_X_advice_no_message_passing_faster(MDP);
%             advise_plot_cmg(MDP);
% 
% 
%     end

   %extract reward
stringsresult = string(subdat.result);
numericresult = str2double(stringsresult);
actualrewards = numericresult(~isnan(numericresult));
actualrewards = actualrewards(:).'; % Reshape into a row

        DCM.trialinfo = trialinfo;
        DCM.field  = field;            % Parameter field
        DCM.U      =  o(:,:);              % trial specification (stimuli)
        DCM.Y      =  u(:,:);              % responses (action)
        DCM.actualrewards = actualrewards;
        DCM.reaction_times = reaction_times;

        DCM.params = params;
        DCM.mode            = 'fit';
        















        DCM        = advice_inversionTT(DCM, model, OMEGAPOSINEGA, MODELBASED);   % Invert the model
        break;
end
     %% 6.3 Check deviation of prior and posterior means & posterior covariance:
        %==========================================================================

        %--------------------------------------------------------------------------
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        fields = fieldnames(DCM.M.pE);
        
        for i = 1:length(fields)
            field = fields{i};
            if ismember(field, {'p_right', 'p_a', 'eta', 'omega', 'eta_a_win', 'omega_a_win',...
                    'eta_a','omega_a','eta_d','omega_d','eta_a_loss','omega_a_loss','eta_d_win',...
                    'omega_d_win', 'eta_d_loss', 'omega_d_loss', 'omega_d_posi', 'omega_d_nega', 'omega_a_posi', 'omega_a_nega', 'lamgda'})
                params.(field) = 1/(1+exp(-DCM.Ep.(field)));
            elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
                    'parameter_exploration', 'Rsensitivity'})
                params.(field) = exp(DCM.Ep.(field));
            else
                params.(field) = DCM.Ep.(field);
            end
        end



        all_MDPs = [];
        %all_MDPs_simmed = [];

        % Simulate beliefs using fitted values
        act_prob_time1=[];
        act_prob_time2 = [];
        model_acc_time1 = [];
        model_acc_time2 = [];

        u = DCM.U;
        y = DCM.Y;



        num_trials = size(u,2);
        num_blocks = floor(num_trials/30);
        if num_trials == 1
            block_size = 1;
        else
            block_size = 30;
        end

        trialinfo = DCM.M.trialinfo;
        L = 0;

        % Each block is separate -- effectively resetting beliefs at the start of
        % each block. 
        for idx_block = 1:num_blocks
            %priors = posteriors;
            %MDP     =
            %advise_gen_model(trialinfo(30*idx_block-29:30*idx_block,:),priors);
            %old model

            if (num_trials == 1)
                outcomes = u;
                actions = y;
                MDP.o  = outcomes{1};
                MDP.u  = actions{1};
                MDP.actualreward  = actualrewards(1);
            else
                outcomes = u(30*idx_block-29:30*idx_block);
                actions  = y(30*idx_block-29:30*idx_block);
                actualreward = actualrewards(30*idx_block-29:30*idx_block);
                for idx_trial = 1:30
                    MDP(idx_trial).o = outcomes{idx_trial};
                    MDP(idx_trial).u = actions{idx_trial};
                    MDP(idx_trial).actualreward = actualreward(idx_trial);
                    MDP(idx_trial).reaction_times = DCM.reaction_times{idx_trial};
                    task.true_p_right(idx_trial) = 1-str2double(trialinfo{(idx_block-1)*30+idx_trial,2});
                    task.true_p_a(idx_trial) = str2double(trialinfo{(idx_block-1)*30+idx_trial,1});
                end
                if strcmp(trialinfo{idx_block*30-29,3}, '80')
                    task.block_type = "LL";
                else
                    task.block_type = "SL";
                end
            end

            % solve MDP and accumulate log-likelihood
            %--------------------------------------------------------------------------

             %MDPs  = spm_MDP_VB_X_advice(MDP); 
             %MDPs  = spm_MDP_VB_X_advice_no_message_passing(MDP); 
             %MDPs  = spm_MDP_VB_X_advice_no_message_passing_faster(MDP); 
             
             %task.field = fields;
             if model == 1
              %MDPs  = spm_MDP_VB_X_advice_no_message_passing_faster(MDP);
              %MDPs  = Simple_Advice_Model_TT(task, MDP, params, 0);
              MDPs  = Simple_Formal_Advice_Model_TT(task, MDP, params, 0);
             elseif model == 2
              MDPs  = ModelFreeRLModelconnect_TT(task, MDP,params, 0);
             elseif model == 3
                 if OMEGAPOSINEGA
                     if MODELBASED
                         MDPs  = ModelBasedRLModeldisconnectPosiNegaForget_TT(task, MDP, params, 0);
                     else
                         MDPs  = ModelFreeRLModeldisconnectPosiNegaForget_TT(task, MDP, params, 0);
                     end
                 else
                     MDPs  = ModelFreeRLModeldisconnect_TT(task, MDP, params, 0);
                 end
             end

             %MDPs_simmed  = ModelFreeRLModeldisconnect_TT(task, MDP,params, 1);
             DCM.action_probs{idx_block} = MDPs.blockwise.action_probs;

             % bandit was chosen
             for j = 1:numel(actions)
                if actions{j}(2,1) ~= 2
                   %act_prob_time1 = [act_prob_time1 MDPs(j).P(1,actions{j}(2,1),1)];
                   action_prob = MDPs.blockwise.action_probs(actions{j}(2,1)-1,1,j);
                   L = L + log(action_prob + eps);
                   act_prob_time1 = [act_prob_time1 action_prob]; 
%                    if MDPs(j).P(1,actions{j}(2,1),1)==max(MDPs(j).P(:,:,1))
%                        model_acc_time1 = [model_acc_time1 1];
%                    else
%                        model_acc_time1 = [model_acc_time1 0];
%                    end
                    if action_prob == max(MDPs.blockwise.action_probs(:,1,j))
                        model_acc_time1 = [model_acc_time1 1];
                    else
                        model_acc_time1 = [model_acc_time1 0];
                    end

                else % when advisor was chosen
                   prob_choose_advisor = MDPs.blockwise.action_probs(1,1,j);
                   L = L + log(prob_choose_advisor + eps);
                   prob_choose_bandit = MDPs.blockwise.action_probs(actions{j}(2,2)-1,2,j);
                   L = L + log(prob_choose_bandit + eps);
                   act_prob_time1 = [act_prob_time1 prob_choose_advisor];
                   act_prob_time2 = [act_prob_time2 prob_choose_bandit];
                   
                   %act_prob_time1 = [act_prob_time1 MDPs(j).P(1,actions{j}(2,1),1)];
                   %act_prob_time2 = [act_prob_time2 MDPs(j).P(1,actions{j}(2,2),2)];
                  % for k = 1:2
%                        if MDPs(j).P(1,actions{j}(2,k),k)==max(MDPs(j).P(:,:,k))
%                            if k ==1
%                               model_acc_time1 = [model_acc_time1 1];
%                            else
%                               model_acc_time2 = [model_acc_time2 1];
%                            end
%                        else
%                            if k ==1
%                               model_acc_time1 = [model_acc_time1 0];
%                            else
%                               model_acc_time2 = [model_acc_time2 0];
%                            end
%                        end
%                   end

                   if prob_choose_advisor==max(MDPs.blockwise.action_probs(:,1,j))
                      model_acc_time1 = [model_acc_time1 1];
                   else
                      model_acc_time1 = [model_acc_time1 0];
                   end
                   if prob_choose_bandit==max(MDPs.blockwise.action_probs(:,2,j))
                      model_acc_time2 = [model_acc_time2 1];
                   else
                      model_acc_time2 = [model_acc_time2 0];
                   end                    
                end
             end
            % Save block of MDPs to list of all MDPs
             all_MDPs = [all_MDPs; MDPs'];
            %all_MDPs_simmed = [all_MDPs_simmed; MDPs_simmed'];

            clear MDPs

        end

        %FinalResults = advise_sim_fitTT(all_MDPs_simmed, field, priors)


        % plotting
        if plot
            % for each trial
            for i=1:length(DCM.U)
                MDP(i).o = DCM.U{1,i};
                MDP(i).u = DCM.Y{1,i};
                MDP(i).reaction_times = DCM.reaction_times{1,i};
                
                block_num = ceil(i/30);
                trial_num_within_block = i - (block_num-1)*30;
                trial_action_probs = all_MDPs(block_num).blockwise.action_probs(:,:,trial_num_within_block);
                % Concatenate the zero row at the top of the matrix
                zero_row = zeros(1, size(trial_action_probs, 2));
                trial_action_probs = vertcat(zero_row, trial_action_probs)';
                MDP(i).P = permute(trial_action_probs, [3 2 1]);
            end
            advise_plot_tt(MDP);

        end
         
         
        fit_results.id = subject;
        fit_results.has_practice_effects = has_practice_effects;
        fit_results.file = file;
        % assign priors/posteriors/fixed params to fit_results
        param_names = fieldnames(params);
        for i = 1:length(param_names)
            % param was fitted
            if ismember(param_names{i}, fields)
                fit_results.(['posterior_' param_names{i}]) = params.(param_names{i});
                fit_results.(['prior_' param_names{i}]) = DCM.params.(param_names{i});  
            % param was fixed
            else
                fit_results.(['fixed_' param_names{i}]) = params.(param_names{i});

            end
        end
        

        fit_results.avg_act_prob_time1 = sum(act_prob_time1)/length(act_prob_time1);
        fit_results.avg_act_prob_time2 = sum(act_prob_time2)/length(act_prob_time2);
        fit_results.avg_model_acc_time1   = sum(model_acc_time1)/length(model_acc_time1);
        fit_results.avg_model_acc_time2   = sum(model_acc_time2)/length(model_acc_time2);
        %fit_results.times_chosen_advisor = length(model_acc_time2);
        fit_results.LL = L;
           
end
