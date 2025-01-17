% Samuel Taylor and Ryan Smith, 2021


function [fit_results, DCM] = advise_sim_fit_uni(subject, folder, sim_data, field, params, plot, model)
    %      load('trialinfo.mat');
    %      num_trials = size(sim_data,1);
    %      trialinfo = trialinfo(1:num_trials,:);
    %      MDP = advise_gen_model(trialinfo, priors);
    %      for bb = 1:size(MDP,2)
    %          if mod(bb, 30) ~= 0
    %              MDP(bb).BN = floor(bb/30)+1;        % block number
    %          else
    %              MDP(bb).BN = floor(bb/30);
    %          end
    %     end
        
        %DCM.MDP    = {sim_data.mdp};                 % MDP model    
            DCM.trialinfo = sim_data(1).trialinfo;
            DCM.field  = field;            % Parameter field
            DCM.U      = sim_data.observations;              % trial specification (stimuli)
            DCM.Y      = sim_data.responses;              % responses (action)
            DCM.actualrewards = sim_data.actualrewards;
            %DCM.reaction_times = reaction_times;
    
            DCM.params = params;
            DCM.mode            = 'fit';
    
    
    
        % Model inversion
        DCM        = advice_inversionTT(DCM, model);   % Invert the model
    
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
                        'omega_d_win', 'eta_d_loss', 'omega_d_loss', 'lamgda'})
                    params.(field) = 1/(1+exp(-DCM.Ep.(field)));
                elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
                        'parameter_exploration','Rsensitivity'})
                    params.(field) = exp(DCM.Ep.(field));
                else
                    params.(field) = DCM.Ep.(field);
                end
            end
       
        
       
        
    
        % Simulate beliefs using fitted values to get avg action prob
        all_MDPs = [];
    
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
                    MDP.actualreward  = sim_data.actualrewards(1);
                else
                    outcomes = u(30*idx_block-29:30*idx_block);
                    actions  = y(30*idx_block-29:30*idx_block);
                    actualreward = sim_data.actualrewards(30*idx_block-29:30*idx_block);
                    for idx_trial = 1:30
                        MDP(idx_trial).o = outcomes{idx_trial};
                        MDP(idx_trial).u = actions{idx_trial};
                        MDP(idx_trial).actualreward = actualreward(idx_trial);
    %                    MDP(idx_trial).reaction_times = DCM.reaction_times{idx_trial};
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
    
                 if model == 1
                  MDPs  = Simple_Advice_Model_TT(task, MDP, params, 0);
                 elseif model == 2
                  MDPs  = ModelFreeRLModelconnect_TT(task, MDP,params, 0);
                 elseif model == 3
                  MDPs  = ModelFreeRLModeldisconnect_TT(task, MDP,params, 0);
                 end
    
     % bandit was chosen
                 for j = 1:numel(actions)
                    if actions{j}(2,1) ~= 2
                       %act_prob_time1 = [act_prob_time1 MDPs(j).P(1,actions{j}(2,1),1)];
                       action_prob = MDPs.blockwise.action_probs(actions{j}(2,1)-1,1,j);
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
                       prob_choose_bandit = MDPs.blockwise.action_probs(actions{j}(2,2)-1,2,j); 
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
     
    % plotting
            if plot
                % for each trial
                for i=1:length(DCM.U)
                    MDP(i).o = DCM.U{1,i};
                    MDP(i).u = DCM.Y{1,i};
    %                MDP(i).reaction_times = DCM.reaction_times{1,i};
                    
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
             
             
            fit_results.idsim = subject;
            %fit_results.has_practice_effects = has_practice_effects;
            fit_results.foldersim = folder;
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
            fit_results.times_chosen_advisor = length(model_acc_time2);   
    
    end