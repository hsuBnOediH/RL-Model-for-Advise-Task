function L = log_likelihood_func(P, M, U, Y),
    % P is free parameters
    % M is model
    q_model = M;
    fields = fieldnames(q_model.pE);
    fixed_fields = fieldnames(q_model.fixed_params);
    preprocessed_data = U;
    % U is input, states
    % Y is response, action and reward
    preprocessed_params = struct();
    % copy everything from P to preprocessed_params
    for i = 1:length(fields)
        preprocessed_params.(fields{i}) = P.(fields{i});
    end
    
    % add the fixed parameters to the params and append the free parameters
    for i = 1:length(fixed_fields)
        preprocessed_params.(fixed_fields{i}) = q_model.fixed_params.(fixed_fields{i});
    end
    % check and transform the range of the parameters
    % get the fields of the P
    fields = fieldnames(preprocessed_params);
    for i = 1:length(fields)
         % for lr and discount_factor range 0-1
         field = fields{i};

         if ismember(field, {'learning_rate', 'with_advice_learning_rate', 'without_advice_learning_rate', 'with_advice_win_learning_rate', 'with_advice_loss_learning_rate', 'without_advice_win_learning_rate', 'without_advice_loss_learning_rate', 'forgetting_rate', 'with_advice_forgetting_rate', 'without_advice_forgetting_rate', 'with_advice_win_forgetting_rate', 'with_advice_loss_forgetting_rate', 'without_advice_win_forgetting_rate', 'without_advice_loss_forgetting_rate', })
            preprocessed_params.(field) = 1/(1+exp(-preprocessed_params.(field)));
        
        elseif ismember(field, {'inv_temp', 'reward_sensitivity', 'loss_sensitivity'})
            preprocessed_params.(field) = log(1+exp(preprocessed_params.(field)));
        end
    end


    params = struct();

    for i = 1:length(fields)
        if strcmp(fields{i},'learning_rate')
            % with advie 
            params.with_advice_win_learning_rate = preprocessed_params.learning_rate;
            params.with_advice_loss_learning_rate = preprocessed_params.learning_rate;
            params.without_advice_win_learning_rate = preprocessed_params.learning_rate;
            params.without_advice_loss_learning_rate = preprocessed_params.learning_rate;
        elseif strcmp(fields{i},'with_advice_learning_rate')
            params.with_advice_win_learning_rate = preprocessed_params.with_advice_learning_rate;
            params.with_advice_loss_learning_rate = preprocessed_params.with_advice_learning_rate;
        elseif strcmp(fields{i},'without_advice_learning_rate')
            params.without_advice_win_learning_rate = preprocessed_params.without_advice_learning_rate;
            params.without_advice_loss_learning_rate = preprocessed_params.without_advice_learning_rate;
        elseif strcmp(fields{i},'with_advice_win_learning_rate')
            params.with_advice_win_learning_rate = preprocessed_params.with_advice_win_learning_rate;
        elseif strcmp(fields{i},'with_advice_loss_learning_rate')
            params.with_advice_loss_learning_rate = preprocessed_params.with_advice_loss_learning_rate;
        elseif strcmp(fields{i},'without_advice_win_learning_rate')
            params.without_advice_win_learning_rate = preprocessed_params.without_advice_win_learning_rate;
        elseif strcmp(fields{i},'without_advice_loss_learning_rate')
            params.without_advice_loss_learning_rate = preprocessed_params.without_advice_loss_learning_rate;
        elseif strcmp(fields{i},'forgetting_rate')
            params.with_advice_win_forgetting_rate = preprocessed_params.forgetting_rate;
            params.with_advice_loss_forgetting_rate = preprocessed_params.forgetting_rate;
            params.without_advice_win_forgetting_rate = preprocessed_params.forgetting_rate;
            params.without_advice_loss_forgetting_rate = preprocessed_params.forgetting_rate;
        elseif strcmp(fields{i},'with_advice_forgetting_rate')
            params.with_advice_win_forgetting_rate = preprocessed_params.with_advice_forgetting_rate;
            params.with_advice_loss_forgetting_rate = preprocessed_params.with_advice_forgetting_rate;
        elseif strcmp(fields{i},'without_advice_forgetting_rate')
            params.without_advice_win_forgetting_rate = preprocessed_params.without_advice_forgetting_rate;
            params.without_advice_loss_forgetting_rate = preprocessed_params.without_advice_forgetting_rate;
        elseif strcmp(fields{i},'with_advice_win_forgetting_rate')
            params.with_advice_win_forgetting_rate = preprocessed_params.with_advice_win_forgetting_rate;
        elseif strcmp(fields{i},'with_advice_loss_forgetting_rate')
            params.with_advice_loss_forgetting_rate = preprocessed_params.with_advice_loss_forgetting_rate;
        elseif strcmp(fields{i},'without_advice_win_forgetting_rate')
            params.without_advice_win_forgetting_rate = preprocessed_params.without_advice_win_forgetting_rate;
        elseif strcmp(fields{i},'without_advice_loss_forgetting_rate')
            params.without_advice_loss_forgetting_rate = preprocessed_params.without_advice_loss_forgetting_rate;
        elseif strcmp(fields{i},'inv_temp')
            params.inv_temp = preprocessed_params.inv_temp;
        elseif strcmp(fields{i},'reward_sensitivity')
            params.reward_sensitivity = preprocessed_params.reward_sensitivity;
        elseif strcmp(fields{i},'loss_sensitivity')
            params.loss_sensitivity = preprocessed_params.loss_sensitivity;
        end
    end
    % Initialize the log likelihood
    L = 0;
    num_trials = length(U);
    action_probs = zeros(num_trials,2,3);

    for i = 1:num_trials
        trial = preprocessed_data(i);
        actual_states = trial.states;
        actual_actions = trial.actions;
        actual_reward = trial.rewards;

        % for time step 1, generate the decison based the Q table and recorad for parameter fitting
        % get first row of Q table, for state 1 and all actions
        q_start_row = q_model.q_table(1,:)* params.inv_temp;
        % softmax the Q table and sample the action
        action_prob_t1 = exp(q_start_row)/sum(exp(q_start_row));
        action_probs(i,1,:) = action_prob_t1;

        % if subject chose advice at time step 1
        if actual_actions(1) == 3
            % determine the which lr to use
            reward_term = 0;
            % reward term is the actual reward times the reward sensitivity or loss sensitivity
            if actual_reward > 0
                reward_term = params.reward_sensitivity * actual_reward;
                lr = params.with_advice_win_learning_rate;
                fr = params.with_advice_win_forgetting_rate;
            else
                reward_term = params.loss_sensitivity * actual_reward;
                lr = params.with_advice_loss_learning_rate;
                fr = params.with_advice_loss_forgetting_rate;
            end
           
            % update the Q table
    
            % new choose(State,Action) = old choose(State,Action) + learning_rate*(reward_term +   max of all furure (state,action) - old choose(State,Action))
            % reward_term is 0 here, because the reward is not known at this time
            % all future state and action are (given advise left, choose left) and (given advise left, choose right), (given advise right, choose left) and (given advise right, choose right)
            q_model.q_table(1,3) = q_model.q_table(1,3) + lr*(max([q_model.q_table(4,1),q_model.q_table(4,2),q_model.q_table(5,1),q_model.q_table(5,2)]) - q_model.q_table(1,3));
            % forget unchosen actions
            % new unchoose(State,Action) = old unchoose(State,Action) * (1-forgetting_rate)
            q_model.q_table(1,1) = q_model.q_table(1,1) * (1-fr);
            q_model.q_table(1,2) = q_model.q_table(1,2) * (1-fr);

        elseif actual_actions(1) == 1
        % if the subject chose left at time step 1

            % determine the which lr to use
            reward_term = 0;
            if actual_reward > 0
                reward_term = params.reward_sensitivity * actual_reward;
                lr = params.without_advice_win_learning_rate;
                fr = params.without_advice_win_forgetting_rate;
            else
                reward_term = params.loss_sensitivity * actual_reward;
                lr = params.without_advice_loss_learning_rate;
                fr = params.without_advice_loss_forgetting_rate;
            end
            % update the Q table
            % here after subject chose left, the trail is over, there is no future state and action
            q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(reward_term  - q_model.q_table(1,1));
            % forget unchosen actions
            q_model.q_table(1,2) = q_model.q_table(1,2) * (1-fr);
            q_model.q_table(1,3) = q_model.q_table(1,3) * (1-fr);
        else
            % if the subject chose right at time step 1
            % determine the which lr to use
            reward_term = 0;
            if actual_reward > 0
                reward_term = params.reward_sensitivity * actual_reward;
                lr = params.without_advice_win_learning_rate;
                fr = params.without_advice_win_forgetting_rate;
            else
                reward_term = params.loss_sensitivity * actual_reward;
                lr = params.without_advice_loss_learning_rate;
                fr = params.without_advice_loss_forgetting_rate;
            end
            % update the Q table
            q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(reward_term - q_model.q_table(1,2));
            % forget unchosen actions
            q_model.q_table(1,1) = q_model.q_table(1,1) * (1-fr);
            q_model.q_table(1,3) = q_model.q_table(1,3) * (1-fr);
        end



        % for time step 2, check the len of the actual actions, countinue for those who have len > 1
        if length(actual_actions) > 1

            second_action = actual_actions(2);
            after_advice_state = actual_states(2);
            % Give the first action is advice, simulate the second action for parameter fitting
            % based on the after_advice_state query the Q table
            q_after_advice_row = q_model.q_table(after_advice_state,1:2)* params.inv_temp;
            % softmax the Q table and sample the action
            action_prob_t2 = exp(q_after_advice_row)/sum(exp(q_after_advice_row));
            action_probs(i,2,1:2) = action_prob_t2;
            

            if second_action == 1
                % if the subject chose left at time step 2
             
                % determine the which lr to use
                reward_term = 0;
                if actual_reward > 0
                    reward_term = params.reward_sensitivity * actual_reward;
                    lr = params.with_advice_win_learning_rate;
                    fr = params.without_advice_win_forgetting_rate;
                else
                    reward_term = params.loss_sensitivity * actual_reward;
                    lr = params.with_advice_loss_learning_rate;
                    fr = params.without_advice_loss_forgetting_rate;
                end
                % update the Q table
            
                q_model.q_table(after_advice_state,1) = q_model.q_table(after_advice_state,1) + lr*(reward_term  - q_model.q_table(after_advice_state,1));
                % forget unchosen actions
                q_model.q_table(after_advice_state,2) = q_model.q_table(after_advice_state,2) * (1-fr);
            else    
                % if the subject chose right
                
                % determine the which lr to use
                if actual_reward > 0
                    reward_term = params.reward_sensitivity * actual_reward;
                    lr = params.with_advice_win_learning_rate;
                    fr = params.without_advice_win_forgetting_rate;
                else
                    reward_term = params.loss_sensitivity * actual_reward;
                    lr = params.with_advice_loss_learning_rate;
                    fr = params.without_advice_loss_forgetting_rate;
                end
                % update the Q table
                q_model.q_table(after_advice_state,2) = q_model.q_table(after_advice_state,2) + lr*(reward_term  - q_model.q_table(after_advice_state,2));
                % forget unchosen actions
                q_model.q_table(after_advice_state,1) = q_model.q_table(after_advice_state,1) * (1-fr);

            end
        end


   

    end 
    

         
    % Calculate the log likelihood
    for trial_idx = 1:num_trials
        actual_actions = preprocessed_data(trial_idx).actions;
        action_prob_t1 = action_probs(trial_idx,1,:);
        action_prob_t2 = action_probs(trial_idx,2,:);
        if length(actual_actions) >1
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = actual_actions(2);
        else
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = 0;
        end
        if actual_action_t1 == 1
            L = L + log(action_prob_t1(1)+eps);
        elseif actual_action_t1 == 2
            L = L + log(action_prob_t1(2)+eps);
        else
            L = L + log(action_prob_t1(3)+eps);
        end
        if actual_action_t2 == 0
            continue
        elseif actual_action_t2 == 1
            L = L + log(action_prob_t2(1)+eps);
        else
            L = L + log(action_prob_t2(2)+eps);
        end
    end        

    fprintf('LL: %f \n',L)
end


