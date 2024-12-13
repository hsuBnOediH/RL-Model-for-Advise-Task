function L = log_likelihood_func(P, M, U, Y)
    % P is free parameters
    % M is model
    q_model = M;
    fields = fieldnames(q_model.pE);
    fixed_fields = fieldnames(q_model.fixed_params);
    preprocessed_data = U;
    % U is input, states,response, action and rewards
    % Y is useless, ignore it

    is_connected = q_model.is_connected;

    % copy the params to preprocessed_params
    preprocessed_params = struct();
    for i = 1:length(fields)
        preprocessed_params.(fields{i}) = P.(fields{i});
    end
    % add the fixed parameters to the params and append the free parameters
    for i = 1:length(fixed_fields)
        preprocessed_params.(fixed_fields{i}) = q_model.fixed_params.(fixed_fields{i});
    end

    % define the fields that need to be transformed
    zero_one_fields = {'left_better','advise_truthness','learning_rate','with_advice_learning_rate','without_advice_learning_rate','with_advice_win_learning_rate','with_advice_loss_learning_rate',...
        'without_advice_win_learning_rate','without_advice_loss_learning_rate','forgetting_rate','with_advice_forgetting_rate','without_advice_forgetting_rate',...
        'with_advice_win_forgetting_rate','with_advice_loss_forgetting_rate','without_advice_win_forgetting_rate','without_advice_loss_forgetting_rate','discount_factor'};
    positive_fields = {'inv_temp','outcome_sensitivity',...
        };

    % check the fields and transform them
    fields = fieldnames(preprocessed_params);
    for i = 1:length(fields)
        field = fields{i};
        if ismember(field, zero_one_fields)
            preprocessed_params.(field) = 1/(1+exp(-preprocessed_params.(field)));
        elseif ismember(field, positive_fields)
            preprocessed_params.(field) = log(1+exp(preprocessed_params.(field)));
        else
            fprintf('The field %s is not in the transformation list\n',field)
        end
    end

    % process the params, replace some fields for generality
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
        elseif strcmp(fields{i},'outcome_sensitivity')
            params.outcome_sensitivity = preprocessed_params.outcome_sensitivity;
        elseif strcmp(fields{i},'large_loss_sensitive')
            params.large_loss_sensitive = preprocessed_params.large_loss_sensitive;
        % this discount factor is for with advise result to update without advise result and vice versa
        elseif strcmp(fields{i},'discount_factor')
            params.discount_factor = preprocessed_params.discount_factor;
        elseif strcmp(fields{i},'left_better')
            params.left_better = preprocessed_params.left_better;
        elseif strcmp(fields{i},'advise_truthness')
            params.advise_truthness = preprocessed_params.advise_truthness;
        end
    end
    % Initialize the log likelihood
    L = 0;
    num_trials = length(U);
    % initialize the action probability for computing the log likelihood
    action_probs = zeros(num_trials,2,3);
    left_better = params.left_better;
    advise_truthness = params.advise_truthness;

    % loop through each trial
    for i = 1:num_trials
        trial = preprocessed_data(i);
        party_size = trial.party_size;

        if party_size == 40
            loss = 4;
        elseif party_size == 80
            loss = params.large_loss_sensitive;
        end

        % if the trial could be divided by the number of trials in one block, reinitialize the Q table
        % need another sensitivity for the largt party size
        if mod(i, 30) == 1
            q_model.q_table = zeros(3,3);
            q_model.q_table(1, 1) = 10 * loss * left_better + (-party_size)* (1-left_better);
            q_model.q_table(1, 2) = 10 * loss * (1-left_better) + (-party_size) * left_better;
            % truthness * value win + (1-truthness) * value lose
            q_model.q_table(1, 3) = 5 * loss * advise_truthness + (-party_size) * (1-advise_truthness);
            q_model.q_table(2, 1) = 5 * loss * advise_truthness + (-party_size) * (1-advise_truthness);
            q_model.q_table(2, 2) = 5 * loss * (1-advise_truthness) + (-party_size) * advise_truthness;
            q_model.q_table(2, 3) = NaN;
            q_model.q_table(3, 1) = 5 * loss * (1-advise_truthness) + (-party_size) * advise_truthness;
            q_model.q_table(3, 2) = 5 * loss * advise_truthness + (-party_size) * (1-advise_truthness);
            q_model.q_table(3, 3) = NaN;
        end
        
        % scale the Q table by the 0.1
        q_model.q_table = q_model.q_table * 0.1;
        start_left_init_q = q_model.q_table(1,1);
        start_right_init_q = q_model.q_table(1,2);
        start_advise_init_q = q_model.q_table(1,3);
        advise_left_left_init_q = q_model.q_table(2,1);
        advise_left_right_init_q = q_model.q_table(2,2);
        advise_right_left_init_q = q_model.q_table(3,1);
        advise_right_right_init_q = q_model.q_table(3,2);

        % element-wise multiplication the outcome sensitivity to the Q table
        outcome_sensitive_table = ones(3,3) * params.outcome_sensitivity;
        q_model.q_table = q_model.q_table .* outcome_sensitive_table;
       
        actual_states = trial.states;
        actual_actions = trial.actions;
        actual_reward = trial.rewards;

        % compute the action probability for each action at time step 1
        % read out the q values for each action at time step 1, time the inv_temp
        q_start_row = q_model.q_table(1,:) * params.inv_temp;
        % softmax the Q table
        action_prob_t1 = exp(q_start_row)/sum(exp(q_start_row));
        action_probs(i,1,:) = action_prob_t1;

        % if subject chose advice at time step 1
        if actual_actions(1) == 3
            % determine the which lr, fr to use
            reward_term = 0;
           
            % current reward sensitivity and loss sensitivity fixed to 1
            if actual_reward > 0
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.with_advice_win_learning_rate;
                fr = params.with_advice_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.with_advice_loss_learning_rate;
                fr = params.with_advice_loss_forgetting_rate;
            end
            % update the Q value for the (start,advise) pair
            % max_future_q = max of (advise_left,left) (advise_left,right) (advise_right,left) (advise_right,right)
            max_future_q = max([q_model.q_table(2,1),q_model.q_table(2,2),q_model.q_table(3,1),q_model.q_table(3,2)]);
            % (start,advise) = (start,advise) + lr * (reward_term + discount_factor * max_future_q - (start,advise))
            % reward_term is 0 here, discount_factor is 1
            % final equation is (start,advise) = (start,advise) + lr * (max_future_q - (start,advise))
            q_model.q_table(1,3) = q_model.q_table(1,3) + lr*(max_future_q - q_model.q_table(1,3));

            % forgetting is not needed for both connected and not connected mode
            

        elseif actual_actions(1) == 1
            % determine the which lr, fr to use
            reward_term = 0;
            if actual_reward > 0
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.without_advice_win_learning_rate;
                fr = params.without_advice_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.without_advice_loss_learning_rate;
                fr = params.without_advice_loss_forgetting_rate;
            end
            % update the Q value for the (start,left) pair
            q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(reward_term  - q_model.q_table(1,1));
            % forget unchosen actions
      
            % might forget for left-right connected, and might not need to forget 
            % init_1_2 = 40 * (1-left_better) + (-party_size) * left_better;
            q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
            q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));

            if is_connected
                % forget for left-right connected
                q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
            end

        elseif actual_actions(1) == 2
            % determine the which lr, fr to use
            reward_term = 0;
            if actual_reward > 0
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.without_advice_win_learning_rate;
                fr = params.without_advice_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.without_advice_loss_learning_rate;
                fr = params.without_advice_loss_forgetting_rate;
            end
            % update the Q value for the (start,right) pair
            q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(reward_term - q_model.q_table(1,2));
            % forget unchosen actions
            q_model.q_table(1,1) = q_model.q_table(1,1) + fr * (start_left_init_q - q_model.q_table(1,1));
            q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));

            if is_connected
                % forget for left-right connected
                q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
            end
        end

        % if for this trial, the subject chose advice and the trial has more than 1 action, update the Q table for the second action and action probability as well
        if length(actual_actions) > 1
            second_action = actual_actions(2);
            after_advice_state = actual_states(2);
            % read out the q values for each action at time step 2, time the inv_temp
            q_after_advice_row = q_model.q_table(after_advice_state,1:2)* params.inv_temp;
            % softmax the Q table and store the action probability
            action_prob_t2 = exp(q_after_advice_row)/sum(exp(q_after_advice_row));
            action_probs(i,2,1:2) = action_prob_t2;


            %  (advise_left,left) if win or loss

            %  learn update
            %  (advise_left,left) = (advise_left,left) + lr * (reward_term - (advise_left,left))

            %  connect update
            %  (advise_rigth,right) = (advise_right,right) + lr * (reward_term - (advise_right,right))
            %  (start,left) = (start,left) + discount_factor * lr * (reward_term - (start,left))

            %  forget unchosen actions, in not connected case
            %  (advise_left,right) = (advise_left,right) + fr * (advise_truthness - (advise_left,right))

            %  (advise_right,left) = (advise_right,left) + fr * (advise_truthness - (advise_right,left))
            %  (start,right) = (start,right) + fr * (advise_truthness - (start,right))

            %  forget unchosen actions, in connected case, not needed

            % advise given is left
            if after_advice_state == 2
                % second action is left
                if second_action == 1
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advice_win_learning_rate;
                        fr = params.without_advice_win_forgetting_rate;
                    else
                        lr = params.with_advice_loss_learning_rate;
                        fr = params.without_advice_loss_forgetting_rate;
                    end
                    % learn update
                    % (advise_left,left) = (advise_left,left) + lr * (reward_term - (advise_left,left))
                    q_model.q_table(2,1) = q_model.q_table(after_advice_state,1) + lr*(reward_term  - q_model.q_table(after_advice_state,1));
                    % learning for without advice
                    % (start,left) = (start,left) + discount_factor * lr * (reward_term - (start,left))
                    q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * lr*(reward_term  - q_model.q_table(1,1));
                    % (advise_rigth,right) = (advise_right,right) + lr * (reward_term - (advise_right,right))
                    q_model.q_table(3,2) = q_model.q_table(3,2) + lr*(reward_term  - q_model.q_table(3,2));

                    % forget update
                    if is_connected
                        % not needed to forget for connected case
                    else
                        %  forget unchosen actions, in not connected case
                        %  (advise_left,right) = (advise_left,right) + fr * (init_value - (advise_left,right))
                        q_model.q_table(2,2) = q_model.q_table(after_advice_state,2) + fr * (advise_left_right_init_q - q_model.q_table(after_advice_state,2));
                        %  (advise_right,left) = (advise_right,left) + fr * (init_value - (advise_right,left))
                        q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                        %  (start,right) = (start,right) + fr * (init_value - (start,right))
                        q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
                    end 

                elseif second_action == 2
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advice_win_learning_rate;
                        fr = params.without_advice_win_forgetting_rate;
                    else
                        lr = params.with_advice_loss_learning_rate;
                        fr = params.without_advice_loss_forgetting_rate;
                    end
                    % learn update
                    % (advise_left,right) = (advise_left,right) + lr * (reward_term - (advise_left,right))
                    q_model.q_table(2,2) = q_model.q_table(after_advice_state,2) + lr*(reward_term  - q_model.q_table(after_advice_state,2));
                    % learning for without advice
                    % (start,right) = (start,right) + discount_factor * lr * (reward_term - (start,right))
                    q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * lr*(reward_term  - q_model.q_table(1,2));
                    % (advise_rigth,left) = (advise_right,left) + lr * (reward_term - (advise_right,left))
                    q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(reward_term  - q_model.q_table(3,1));

                    % forget update
                    if is_connected
                        % not needed to forget for connected case
                    else
                        %  forget unchosen actions, in not connected case
                        %  (advise_left,left) = (advise_left,left) + fr * (init_value - (advise_left,left))
                        q_model.q_table(2,1) = q_model.q_table(after_advice_state,1) + fr * (advise_left_left_init_q - q_model.q_table(after_advice_state,1));
                        %  (advise_right,right) = (advise_right,right) + fr * (init_value - (advise_right,right))
                        q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
                        %  (start,left) = (start,left) + fr * (init_value - (start,left))
                        q_model.q_table(1,1) = q_model.q_table(1,1) + fr * (start_left_init_q - q_model.q_table(1,1));
                    end
                
                else
                    % warning
                    fprintf('The second action is not 1 or 2\n')

                end
            % advise given is right
            elseif after_advice_state == 3
                if second_action == 1
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advice_win_learning_rate;
                        fr = params.without_advice_win_forgetting_rate;
                    else
                        lr = params.with_advice_loss_learning_rate;
                        fr = params.without_advice_loss_forgetting_rate;
                    end
                    % learn update
                    % (advise_right,left) = (advise_right,left) + lr * (reward_term - (advise_right,left))
                    q_model.q_table(3,1) = q_model.q_table(after_advice_state,1) + lr*(reward_term  - q_model.q_table(after_advice_state,1));
                    % learning for without advice
                    % (start,left) = (start,left) + discount_factor * lr * (reward_term - (start,left))
                    q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * lr*(reward_term  - q_model.q_table(1,1));
                    % (advise_left,right) = (advise_left,right) + lr * (reward_term - (advise_left,right))
                    q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(reward_term  - q_model.q_table(2,2));

                    % forget update
                    if is_connected
                        % not needed to forget for connected case
                    else
                        %  forget unchosen actions, in not connected case
                        %  (advise_right,right) = (advise_right,right) + fr * (init_value - (advise_right,right))
                        q_model.q_table(3,2) = q_model.q_table(after_advice_state,2) + fr * (advise_right_right_init_q - q_model.q_table(after_advice_state,2));
                        %  (advise_left,left) = (advise_left,left) = fr * (init_value - (advise_left,left))
                        q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                        %  (start,right) = (start,right) = fr * (init_value - (start,right))
                        q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
                    end

                elseif second_action == 2
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advice_win_learning_rate;
                        fr = params.without_advice_win_forgetting_rate;
                    else
                        lr = params.with_advice_loss_learning_rate;
                        fr = params.without_advice_loss_forgetting_rate;
                    end
                    % learn update
                    % (advise_right,right) = (advise_right,right) + lr * (reward_term - (advise_right,right))
                    q_model.q_table(3,2) = q_model.q_table(after_advice_state,2) + lr*(reward_term  - q_model.q_table(after_advice_state,2));
                    % learning for without advice
                    % (start,right) = (start,right) + discount_factor * lr * (reward_term - (start,right))
                    q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * lr*(reward_term  - q_model.q_table(1,2));
                    % (advise_left,left) = (advise_left,left) = lr * (reward_term - (advise_left,left))
                    q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(reward_term  - q_model.q_table(2,1));

                    % forget update
                    if is_connected
                        % not needed to forget for connected case
                    else
                        %  forget unchosen actions, in not connected case
                        %  (advise_right,left) = (advise_right,left) = fr * (init_value - (advise_right,left))
                        q_model.q_table(3,1) = q_model.q_table(after_advice_state,1) + fr * (advise_right_left_init_q - q_model.q_table(after_advice_state,1));
                        %  (advise_left,right) = (advise_left,right) = fr * (init_value - (advise_left,right))
                        q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                        %  (start,left) = (start,left) = fr * (init_value - (start,left))
                        q_model.q_table(1,1) = q_model.q_table(1,1) + fr * (start_left_init_q - q_model.q_table(1,1));
                    end
                
                else
                    % warning
                    fprintf('The second action is not 1 or 2\n')

                end
            else
                % warning
                fprintf('The advise state is not 2 or 3\n')
            end
    end
         
    % Calculate the log likelihood using the action probabilities
    for trial_idx = 1:num_trials
        actual_actions = preprocessed_data(trial_idx).actions;
        action_prob_t1 = action_probs(trial_idx,1,:);
        action_prob_t2 = action_probs(trial_idx,2,:);
        % read out the actual actions at time step 1 and 2 depending on the length of the actual actions
        if length(actual_actions) >1
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = actual_actions(2);
        else
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = 0;
        end
        % compute the log likelihood for each action at time step 1 
        if actual_action_t1 == 1
            L = L + log(action_prob_t1(1)+eps);
        elseif actual_action_t1 == 2
            L = L + log(action_prob_t1(2)+eps);
        else
            L = L + log(action_prob_t1(3)+eps);
        end
        % compute the log likelihood for each action at time step 2
        if actual_action_t2 == 0
            continue
        elseif actual_action_t2 == 1
            L = L + log(action_prob_t2(1)+eps);
        elseif actual_action_t2 == 2
            L = L + log(action_prob_t2(2)+eps);
        end
    end        

    fprintf('LL: %f \n',L)
end


