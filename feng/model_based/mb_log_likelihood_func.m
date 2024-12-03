function L = mb_log_likelihood_func(P, M, U, Y),
    % P is free parameters
    % M is model
    mb_model = M;
    fields = fieldnames(mb_model.pE);
    fixed_fields = fieldnames(mb_model.fixed_params);
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
        preprocessed_params.(fixed_fields{i}) = mb_model.fixed_params.(fixed_fields{i});
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
        % this discount factor is for with advise result to update without advise result and vice versa
        elseif strcmp(fields{i},'discount_factor')
            params.discount_factor = preprocessed_params.discount_factor;
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
        actual_party_size = trial.party_size;
        is_win = actual_reward > 0;

        % to get the action probability of first time step
        action_prob_t1 = zeros(1,3);
        % 1. query the prob table for the first state
        prob_start_row = mb_model.prob_table(1,:);
        prob_left_advice_row= mb_model.prob_table(2,:);
        prob_right_advice_row = mb_model.prob_table(3,:);
        % 2. compute the expected reward for each action
        % 2.1 chose left
        chose_left_win_prob = prob_start_row(1);
        chose_left_expected_reward = chose_left_win_prob * 40 + (1-chose_left_win_prob) * (-actual_party_size);
        action_prob_t1(1) = chose_left_expected_reward;
        % 2.2 chose right
        chose_right_win_prob = prob_start_row(2);
        chose_right_expected_reward = chose_right_win_prob * 40 + (1-chose_right_win_prob) * (-actual_party_size);
        action_prob_t1(2) = chose_right_expected_reward;
        % 2.3 take advice
        take_left_advice_prob = prob_start_row(3);
        take_right_advice_prob = 1- take_left_advice_prob;
        take_left_advise_chosen_left_win_prob = prob_left_advice_row(1);
        take_left_advise_chosen_right_win_prob = prob_left_advice_row(2);

        take_right_advise_chosen_left_win_prob = prob_right_advice_row(1);
        take_right_advise_chosen_right_win_prob = prob_right_advice_row(2);

        take_left_adivse_expected_reward = take_left_advise_chosen_left_win_prob * 20 + (1-take_left_advise_chosen_left_win_prob) * (-actual_party_size) + take_left_advise_chosen_right_win_prob * 20 + (1-take_left_advise_chosen_right_win_prob) * (-actual_party_size);
        take_right_advise_expected_reward = take_right_advise_chosen_left_win_prob * 20 + (1-take_right_advise_chosen_left_win_prob) * (-actual_party_size) + take_right_advise_chosen_right_win_prob * 20 + (1-take_right_advise_chosen_right_win_prob) * (-actual_party_size);

        take_advise_expected_reward = take_left_adivse_expected_reward * take_left_advice_prob + take_right_advise_expected_reward * take_right_advice_prob;

        action_prob_t1(3) = take_advise_expected_reward;
        % 3. time the probs with the inverse temperature
        action_prob_t1 = action_prob_t1 * params.inv_temp;
        % 4. softmax the probs to get the action probability
        action_prob_t1 = exp(action_prob_t1)/sum(exp(action_prob_t1));
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
           
            % update the mb_model prob table
            % if the second states is advise left
            is_left_advice = actual_states(2) == 2;
    
            mb_model.prob_table(1,3) = mb_model.prob_table(1,3) + lr*(is_left_advice - mb_model.prob_table(1,3));

            % forget unchosen actions !!! HOW TO FORGET UNCHOSEN ACTIONS FOR THIS MODEL?
            % new unchoose(State,Action) = old unchoose(State,Action) * (1-forgetting_rate)
            mb_model.prob_table(1,1) = mb_model.prob_table(1,1) * (1-fr);
            mb_model.prob_table(1,2) = mb_model.prob_table(1,2) * (1-fr);
            

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
           

            mb_model.prob_table(1,1) = mb_model.prob_table(1,1) + lr*(is_win - mb_model.prob_table(1,1));        
            % forget unchosen actions
            mb_model.prob_table(1,2) = mb_model.prob_table(1,2) * (1-fr);
            mb_model.prob_table(1,3) = mb_model.prob_table(1,3) * (1-fr);
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

            mb_model.prob_table(1,2) = mb_model.prob_table(1,2) + lr*(is_win - mb_model.prob_table(1,2));
            % forget unchosen actions
            mb_model.prob_table(1,1) = mb_model.prob_table(1,1) * (1-fr);
            mb_model.prob_table(1,3) = mb_model.prob_table(1,3) * (1-fr);
        end



        % for time step 2, check the len of the actual actions, countinue for those who have len > 1
        if length(actual_actions) > 1

            second_action = actual_actions(2);
            after_advice_state = actual_states(2);
            
            if after_advice_state == 2
                % left advice
                after_advice_state = 2;
            else
                % right advice
                after_advice_state = 3;
            end


            action_prob_t2 = zeros(1,2);
            after_advice_prob_row = mb_model.prob_table(after_advice_state,1:2);

            after_advice_chose_left_win_prob = after_advice_prob_row(1);
            after_advice_chose_right_win_prob = after_advice_prob_row(2);

            expected_reward_left = after_advice_chose_left_win_prob * 20 + (1-after_advice_chose_left_win_prob) * (-actual_party_size);
            expected_reward_right = after_advice_chose_right_win_prob * 20 + (1-after_advice_chose_right_win_prob) * (-actual_party_size);

            action_prob_t2(1) = expected_reward_left;
            action_prob_t2(2) = expected_reward_right;

            % softmax the Q table and sample the action
            action_prob_t2 = exp(action_prob_t2)/sum(exp(action_prob_t2));
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
            
                mb_model.prob_table(after_advice_state,1) = mb_model.prob_table(after_advice_state,1) + lr*(is_win - mb_model.prob_table(after_advice_state,1));
                % update the without advice Q table based on the with advice Q table
                mb_model.prob_table(1,1) = mb_model.prob_table(1,1) + params.discount_factor * lr*(is_win  - mb_model.prob_table(1,1));
                % forget unchosen actions
                mb_model.prob_table(after_advice_state,2) = mb_model.prob_table(after_advice_state,2) * (1-fr);
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
                mb_model.prob_table(after_advice_state,2) = mb_model.prob_table(after_advice_state,2) + lr*(is_win - mb_model.prob_table(after_advice_state,2));
                % update the without advice Q table based on the with advice Q table
                mb_model.prob_table(1,2) = mb_model.prob_table(1,2) + params.discount_factor * lr*(is_win  - mb_model.prob_table(1,2));
                % forget unchosen actions
                mb_model.prob_table(after_advice_state,1) = mb_model.prob_table(after_advice_state,1) * (1-fr);

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


