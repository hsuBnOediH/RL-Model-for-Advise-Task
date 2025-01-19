function L = log_likelihood_func(P, M, U, Y)
    % P is free parameters
    % M is model
    q_model = M;
    fields = fieldnames(P);
    fixed_fields = fieldnames(q_model.fixed_params);
    preprocessed_data = U;
    % U is input, states,response, action and rewards
    % Y is useless, ignore it

    is_connected = q_model.is_connected;

    % define the fields that need to be transformed
    zero_one_fields = M.zero_one_fields;
    positive_fields = M.positive_fields;
   
    % copy the params to preprocessed_params
    preprocessed_params = struct();
    for i = 1:length(fields)

        field = fields{i};
        if ismember(field, zero_one_fields)
            preprocessed_params.(field) = 1/(1+exp(-P.(field)));
        elseif ismember(field, positive_fields)
            preprocessed_params.(field) = exp(P.(field));
        end


    end
    % add the fixed parameters to the params and append the free parameters
    for i = 1:length(fixed_fields)
        preprocessed_params.(fixed_fields{i}) = q_model.fixed_params.(fixed_fields{i});
    end

 
    % check the fields and transform them
    fields = fieldnames(preprocessed_params);
  

    % process the params, replace some fields for generality
    params = struct();

    for i = 1:length(fields)
        if strcmp(fields{i},'learning_rate')
            % with advie 
            params.with_advise_win_learning_rate = preprocessed_params.learning_rate;
            params.with_advise_loss_learning_rate = preprocessed_params.learning_rate;
            params.without_advise_win_learning_rate = preprocessed_params.learning_rate;
            params.without_advise_loss_learning_rate = preprocessed_params.learning_rate;
        elseif strcmp(fields{i},'with_advise_learning_rate')
            params.with_advise_win_learning_rate = preprocessed_params.with_advise_learning_rate;
            params.with_advise_loss_learning_rate = preprocessed_params.with_advise_learning_rate;
        elseif strcmp(fields{i},'without_advise_learning_rate')
            params.without_advise_win_learning_rate = preprocessed_params.without_advise_learning_rate;
            params.without_advise_loss_learning_rate = preprocessed_params.without_advise_learning_rate;
        elseif strcmp(fields{i},'with_advise_win_learning_rate')
            params.with_advise_win_learning_rate = preprocessed_params.with_advise_win_learning_rate;
        elseif strcmp(fields{i},'with_advise_loss_learning_rate')
            params.with_advise_loss_learning_rate = preprocessed_params.with_advise_loss_learning_rate;
        elseif strcmp(fields{i},'without_advise_win_learning_rate')
            params.without_advise_win_learning_rate = preprocessed_params.without_advise_win_learning_rate;
        elseif strcmp(fields{i},'without_advise_loss_learning_rate')
            params.without_advise_loss_learning_rate = preprocessed_params.without_advise_loss_learning_rate;
        elseif strcmp(fields{i},'forgetting_rate')
            params.with_advise_win_forgetting_rate = preprocessed_params.forgetting_rate;
            params.with_advise_loss_forgetting_rate = preprocessed_params.forgetting_rate;
            params.without_advise_win_forgetting_rate = preprocessed_params.forgetting_rate;
            params.without_advise_loss_forgetting_rate = preprocessed_params.forgetting_rate;
        elseif strcmp(fields{i},'with_advise_forgetting_rate')
            params.with_advise_win_forgetting_rate = preprocessed_params.with_advise_forgetting_rate;
            params.with_advise_loss_forgetting_rate = preprocessed_params.with_advise_forgetting_rate;
        elseif strcmp(fields{i},'without_advise_forgetting_rate')
            params.without_advise_win_forgetting_rate = preprocessed_params.without_advise_forgetting_rate;
            params.without_advise_loss_forgetting_rate = preprocessed_params.without_advise_forgetting_rate;
        elseif strcmp(fields{i},'with_advise_win_forgetting_rate')
            params.with_advise_win_forgetting_rate = preprocessed_params.with_advise_win_forgetting_rate;
        elseif strcmp(fields{i},'with_advise_loss_forgetting_rate')
            params.with_advise_loss_forgetting_rate = preprocessed_params.with_advise_loss_forgetting_rate;
        elseif strcmp(fields{i},'without_advise_win_forgetting_rate')
            params.without_advise_win_forgetting_rate = preprocessed_params.without_advise_win_forgetting_rate;
        elseif strcmp(fields{i},'without_advise_loss_forgetting_rate')
            params.without_advise_loss_forgetting_rate = preprocessed_params.without_advise_loss_forgetting_rate;
        elseif strcmp(fields{i},'inv_temp')
            params.inv_temp = preprocessed_params.inv_temp;
        elseif strcmp(fields{i},'outcome_sensitivity')
            params.outcome_sensitivity = preprocessed_params.outcome_sensitivity;
        elseif strcmp(fields{i},'r_sensitivity')
            params.r_sensitivity = preprocessed_params.r_sensitivity;
        elseif strcmp(fields{i},'large_loss_sensitive')
            params.large_loss_sensitive = preprocessed_params.large_loss_sensitive;
        % this discount factor is for with advise result to update without advise result and vice versa
        elseif strcmp(fields{i},'discount_factor')
            params.discount_factor = preprocessed_params.discount_factor;
        elseif strcmp(fields{i},'left_better')
            params.left_better = preprocessed_params.left_better;
        elseif strcmp(fields{i},'advise_truthness')
            params.advise_truthness = preprocessed_params.advise_truthness;
        elseif strcmp(fields{i},'reaction_time_threshold')
            params.reaction_time_threshold = preprocessed_params.reaction_time_threshold;
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
            loss = params.large_loss_sensitive;
        elseif party_size == 80
            loss = params.large_loss_sensitive * params.r_sensitivity;
        end


        % if the trial could be divided by the number of trials in one block, reinitialize the Q table
        % need another sensitivity for the largt party size
        if mod(i, 30) == 1
            q_model.q_table = zeros(3,3);
            q_model.q_table(1, 1) = 4 * left_better -(loss* (1-left_better)) ;
            q_model.q_table(1, 2) = 4 * (1-left_better) - loss * left_better;
            % truthness * value win + (1-truthness) * value lose
            q_model.q_table(1, 3) = 2 * advise_truthness - loss * (1-advise_truthness);
            q_model.q_table(2, 1) = 2 * advise_truthness -loss * (1-advise_truthness);
            q_model.q_table(2, 2) = 2 * (1-advise_truthness) -loss * advise_truthness;

            q_model.q_table(3, 1) = 2 * (1-advise_truthness) -loss * advise_truthness;
            q_model.q_table(3, 2) = 2 * advise_truthness -loss * (1-advise_truthness);

            outcome_sensitive_table = ones(3,3) * params.outcome_sensitivity;
            q_model.q_table = q_model.q_table .* outcome_sensitive_table;

            start_left_init_q = q_model.q_table(1,1);
            start_right_init_q = q_model.q_table(1,2);
            start_advise_init_q = q_model.q_table(1,3);
            advise_left_left_init_q = q_model.q_table(2,1);
            advise_left_right_init_q = q_model.q_table(2,2);
            advise_right_left_init_q = q_model.q_table(3,1);
            advise_right_right_init_q = q_model.q_table(3,2);
        end
 
        



        % element-wise multiplication the outcome sensitivity to the Q table    
        actual_states = trial.states;
        actual_actions = trial.actions;
        % scale the rewards by 0.1 times, if is loss, use the loss value init by the party size
        actual_reward = trial.rewards;
        if actual_reward > 0
            actual_reward = actual_reward * 0.1;
        else
            actual_reward = -loss;
        end
        

        % compute the action probability for each action at time step 1
        % read out the q values for each action at time step 1, time the inv_temp
        q_start_row = q_model.q_table(1,:) * params.inv_temp;
        q_start_row = reshape(q_start_row, [], 1);
        % softmax the Q table
        action_prob_t1 = spm_softmax(q_start_row);
        action_probs(i,1,:) = action_prob_t1;

        % if subject chose advise at time step 1
        if actual_actions(1) == 3
            % determine the which lr, fr to use
            reward_term = 0;
            % current reward sensitivity and loss sensitivity fixed to 1
            if actual_reward > 0
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.with_advise_win_learning_rate;
                fr = params.with_advise_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * actual_reward;
                lr = params.with_advise_loss_learning_rate;
                fr = params.with_advise_loss_forgetting_rate;
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
                opposite_reward = params.outcome_sensitivity * (-loss);
                lr = params.without_advise_win_learning_rate;
                fr = params.with_advise_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * (-loss);
                opposite_reward = params.outcome_sensitivity * 4;
                lr = params.without_advise_loss_learning_rate;
                fr = params.with_advise_loss_forgetting_rate;
            end
            % for case of (start,left)
            %   connect version:
            %       leran update: (start,left), (start,right)
            %       forget update: (advise_left,left), (advise_left,right), (advise_right,left), (advise_right,right), (start,advise)
            %   not connect version:
            %       learn update: (start,left)
            %       forget update: (start,right), (advise_left,left), (advise_left,right), (advise_right,left), (advise_right,right), (start,advise)
            if is_connected
                % learn update

                q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(reward_term  - q_model.q_table(1,1));
                q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(opposite_reward  - q_model.q_table(1,2));
                % forget update
                q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
                q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));
            else
                % learn update
                q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(reward_term  - q_model.q_table(1,1));
                % forget update
                q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
                q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));
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
                opposite_reward = params.outcome_sensitivity * (-loss);
                lr = params.without_advise_win_learning_rate;
                fr = params.with_advise_win_forgetting_rate;
            else
                reward_term = params.outcome_sensitivity * actual_reward;
                opposite_reward = params.outcome_sensitivity * 4;
                lr = params.without_advise_loss_learning_rate;
                fr = params.without_advise_loss_forgetting_rate;
            end
            % for case of (start,right)
            %  connect version:
            %      learn update: (start,left), (start,right)
            %      forget update: (advise_left,left), (advise_left,right), (advise_right,left), (advise_right,right), (start,advise)
            %  not connect version:
            %      learn update: (start,right)
            %      forget update: (start,left), (advise_left,left), (advise_left,right), (advise_right,left), (advise_right,right), (start,advise)

            if is_connected
                % learn update
                
                q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(reward_term  - q_model.q_table(1,2));
                q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(opposite_reward  - q_model.q_table(1,1));
                % forget update
                q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
                q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));
            else
                % learn update
                q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(reward_term  - q_model.q_table(1,2));
                % forget update
                q_model.q_table(1,1) = q_model.q_table(1,1) + fr * (start_left_init_q - q_model.q_table(1,1));
                q_model.q_table(1,3) = q_model.q_table(1,3) + fr * (start_advise_init_q - q_model.q_table(1,3));
                q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
            end   
        end

        % if for this trial, the subject chose advise and the trial has more than 1 action, update the Q table for the second action and action probability as well
        if length(actual_actions) > 1
            second_action = actual_actions(2);
            after_advise_state = actual_states(2);
            % read out the q values for each action at time step 2, time the inv_temp
            q_after_advise_row = q_model.q_table(after_advise_state,1:2)* params.inv_temp;
            q_after_advise_row = reshape(q_after_advise_row, [], 1);
            % softmax the Q table and store the action probability
            action_prob_t2 = spm_softmax(q_after_advise_row);
            action_probs(i,2,1:2) = action_prob_t2;

            if actual_reward > 0
                    reward_term = 2  * params.outcome_sensitivity;
                    opposite_reward = (-loss) * params.outcome_sensitivity;
                    without_advise_actual_reward = 4 * params.outcome_sensitivity;
                    without_advise_opposite_reward = (-loss) * params.outcome_sensitivity;
                
            else
                    reward_term = (-loss) * params.outcome_sensitivity;
                    opposite_reward = 2 * params.outcome_sensitivity;
                    without_advise_actual_reward = (-loss) * params.outcome_sensitivity;
                    without_advise_opposite_reward = 4 * params.outcome_sensitivity;
                
            end


            if after_advise_state == 2
                % second action is left
                if second_action == 1
                   % determine the which lr, fr to use
                    if actual_reward > 0
                        lr = params.with_advise_win_learning_rate;
                        fr = params.without_advise_win_forgetting_rate;
                        wo_advise_lr = params.without_advise_win_learning_rate;             
                    else
                        lr = params.with_advise_loss_learning_rate;
                        fr = params.without_advise_loss_forgetting_rate;
                        wo_advise_lr = params.without_advise_loss_learning_rate;
                    end


                    % for case of (advise_left,left)
                    %  connect version:
                    %      learn update: (advise_left,left), (advise_left,right), (start,left), (start,right), (advise_right,right), (advise_right,left), (advise_left,right), (advise_right,left)
                    %      forget update: None
                    %  not connect version:
                    %      learn update: (advise_left,left), (advise_right,right), (start,left)
                    %      forget update: (advise_left,right), (advise_right,left), (start,right)
                    % advise given is left
                    
                    if is_connected
                        % learn update
                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(reward_term  - q_model.q_table(2,1));
                        q_model.q_table(3,2) = q_model.q_table(3,2) + lr*(reward_term  - q_model.q_table(3,2));

                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(opposite_reward  - q_model.q_table(2,2));
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(opposite_reward  - q_model.q_table(3,1));

                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,1));
                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_opposite_reward  - q_model.q_table(1,2));
                        % forget update None
                    else
                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(reward_term  - q_model.q_table(2,1));
                        q_model.q_table(3,2) = q_model.q_table(3,2) + lr*(reward_term  - q_model.q_table(3,2));

                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,1));

                        % forget update
                        q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
                        q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                        q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
                    end

                elseif second_action == 2
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advise_win_learning_rate;
                        fr = params.without_advise_win_forgetting_rate;
                        wo_advise_lr = params.without_advise_win_learning_rate;
                    else
                        lr = params.with_advise_loss_learning_rate;
                        fr = params.without_advise_loss_forgetting_rate;
                        wo_advise_lr = params.without_advise_loss_learning_rate;
                    end
                    % for case of (advise_left,right)
                    %  connect version:
                    %      learn update: (advise_left,left), (advise_left,right), (start,left), (start,right), (advise_right,right), (advise_right,left), (advise_left,right), (advise_right,left)
                    %      forget update: None
                    %  not connect version:
                    %      learn update: (advise_left,right), (advise_right,left), (start,right)
                    %      forget update: (advise_left,left), (advise_right,right), (start,left)

                    if is_connected
                        % learn update
                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(reward_term  - q_model.q_table(2,2));
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(reward_term  - q_model.q_table(3,1));

                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(opposite_reward  - q_model.q_table(2,1));
                        q_model.q_table(3,2) = q_model.q_table(3,2) + lr*(opposite_reward  - q_model.q_table(3,2));

                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,2));
                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_opposite_reward  - q_model.q_table(1,1));
                        % forget update None
                    else
                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(reward_term  - q_model.q_table(2,2));
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(reward_term  - q_model.q_table(3,1));

                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,2));

                        % forget update
                        q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                        q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
                        q_model.q_table(1,1) = q_model.q_table(1,1) + fr * (start_left_init_q - q_model.q_table(1,1));
                    end
                else
                    % warning
                    fprintf('The second action is not 1 or 2\n')
                end
            % advise given is right
            elseif after_advise_state == 3
                if second_action == 1
                    % determine the which lr, fr to use
                    reward_term = params.outcome_sensitivity * actual_reward;
                    if actual_reward > 0
                        lr = params.with_advise_win_learning_rate;
                        fr = params.without_advise_win_forgetting_rate;
                        wo_advise_lr = params.without_advise_win_learning_rate;
                    else
                        lr = params.with_advise_loss_learning_rate;
                        fr = params.without_advise_loss_forgetting_rate;
                        wo_advise_lr = params.without_advise_loss_learning_rate;
                    end

                    % for case of (advise_right,left)
                    %  connect version:
                    %      learn update: (advise_left,left), (advise_left,right), (start,left), (start,right), (advise_right,right), (advise_right,left), (advise_left,right), (advise_right,left)
                    %      forget update: None
                    %  not connect version:
                    %      learn update: (advise_right,left), (advise_left,right), (start,left)
                    %      forget update: (advise_right,right), (advise_left,left), (start,right)

                    if is_connected
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(reward_term  - q_model.q_table(3,1));
                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(reward_term  - q_model.q_table(2,2));

                        q_model.q_table(3,2) = q_model.q_table(3,2) + lr*(opposite_reward  - q_model.q_table(3,2));
                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(opposite_reward  - q_model.q_table(2,1));

                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,1));
                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_opposite_reward  - q_model.q_table(1,2));
                        % forget update None
                    else
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(reward_term  - q_model.q_table(3,1));
                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(reward_term  - q_model.q_table(2,2));

                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,1));

                        % forget update
                        q_model.q_table(3,2) = q_model.q_table(3,2) + fr * (advise_right_right_init_q - q_model.q_table(3,2));
                        q_model.q_table(2,1) = q_model.q_table(2,1) + fr * (advise_left_left_init_q - q_model.q_table(2,1));
                        q_model.q_table(1,2) = q_model.q_table(1,2) + fr * (start_right_init_q - q_model.q_table(1,2));
                    end
                elseif second_action == 2
                    if actual_reward > 0
                        lr = params.with_advise_win_learning_rate;
                        fr = params.without_advise_win_forgetting_rate;
                        wo_advise_lr = params.without_advise_win_learning_rate;
                    else
                        lr = params.with_advise_loss_learning_rate;
                        fr = params.without_advise_loss_forgetting_rate;
                        wo_advise_lr = params.without_advise_loss_learning_rate;
                    end

                    % for case of (advise_right,right)
                    %  connect version:
                    %      learn update: (advise_left,left), (advise_left,right), (start,left), (start,right), (advise_right,right), (advise_right,left), (advise_left,right), (advise_right,left)
                    %      forget update: None

                    %  not connect version:
                    %      learn update: (advise_right,right), (advise_left,left), (start,right)
                    %      forget update: (advise_right,left), (advise_left,right), (start,left)
                    if is_connected
                        q_model.q_table(3,2) = q_model.q_table(after_advise_state,2) + lr*(reward_term  - q_model.q_table(after_advise_state,2));
                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(reward_term  - q_model.q_table(2,1));
                        q_model.q_table(3,1) = q_model.q_table(3,1) + lr*(opposite_reward  - q_model.q_table(3,1));
                        q_model.q_table(2,2) = q_model.q_table(2,2) + lr*(opposite_reward  - q_model.q_table(2,2));

                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,2));
                        q_model.q_table(1,1) = q_model.q_table(1,1) + params.discount_factor * wo_advise_lr*(without_advise_opposite_reward  - q_model.q_table(1,1));

                        % forget update None
                    else
                        q_model.q_table(3,2) = q_model.q_table(after_advise_state,2) + lr*(reward_term  - q_model.q_table(after_advise_state,2));
                        q_model.q_table(2,1) = q_model.q_table(2,1) + lr*(reward_term  - q_model.q_table(2,1));
                        
                        q_model.q_table(1,2) = q_model.q_table(1,2) + params.discount_factor * wo_advise_lr*(without_advise_actual_reward  - q_model.q_table(1,2));
                        % forget update
                        q_model.q_table(3,1) = q_model.q_table(3,1) + fr * (advise_right_left_init_q - q_model.q_table(3,1));
                        q_model.q_table(2,2) = q_model.q_table(2,2) + fr * (advise_left_right_init_q - q_model.q_table(2,2));
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
    end
         
    % Calculate the log likelihood using the action probabilities
    for trial_idx = 1:num_trials
        reaction_times = preprocessed_data(trial_idx).reaction_time;
        actual_actions = preprocessed_data(trial_idx).actions;
        action_prob_t1 = action_probs(trial_idx,1,:);
        action_prob_t2 = action_probs(trial_idx,2,:);
        % read out the actual actions at time step 1 and 2 depending on the length of the actual actions
        if length(actual_actions) >1
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = actual_actions(2);
            
            reaction_time_t1 = reaction_times(1);
            reaction_time_t2 = reaction_times(2);
        else
            actual_action_t1 = actual_actions(1);
            actual_action_t2 = 0;
            reaction_time_t1 = 0;
            reaction_time_t2 = reaction_times(1);
        end

        % skip the log likelihood computation if the reaction time is larger than N seconds
       
        if reaction_time_t1 < params.reaction_time_threshold
            % compute the log likelihood for each action at time step 1 
            if actual_action_t1 == 1
                L = L + log(action_prob_t1(1)+eps);
            elseif actual_action_t1 == 2
                L = L + log(action_prob_t1(2)+eps);
            else
                L = L + log(action_prob_t1(3)+eps);
            end
        end
        % compute the log likelihood for each action at time step 2
        if reaction_time_t2 < params.reaction_time_threshold
                
            if actual_action_t2 == 0
                continue
            elseif actual_action_t2 == 1
                L = L + log(action_prob_t2(1)+eps);
            elseif actual_action_t2 == 2
                L = L + log(action_prob_t2(2)+eps);
            end
        end
    end        

    fprintf('LL: %f \n',L)
end


