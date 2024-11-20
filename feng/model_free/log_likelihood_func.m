function L = log_likelihood_func(P, M, U, Y),
    % P is free parameters
    % M is model
    q_model = M;
    fields = fieldnames(q_model.pE);
    preprocessed_data = U;
    % U is input, states
    % Y is response, action and reward
    params = struct();

    % check and transform the range of the parameters
    % get the fields of the P
    fields = fieldnames(P);
    for i = 1:length(fields)
         % for lr and discount_factor range 0-1
         field = fields{i};

         if ismember(field, {'lr','lr_advice','lr_self','lr_left','lr_right','lr_win','lr_loss','discount_factor'})
            P.(field) = 1/(1+exp(-P.(field)));
        
        elseif ismember(field, {'inv_temp'})
            P.(field) = log(1+exp(P.(field)));
        end
    end


    for i = 1:length(fields)
        if strcmp(fields{i},'lr')
            % with advie 
            params.lr_advice_left_win = P.lr;
            params.lr_advice_left_loss = P.lr;
            params.lr_advice_right_win = P.lr;
            params.lr_advice_right_loss = P.lr;
            % without advice
            params.lr_self_left_win = P.lr;
            params.lr_self_left_loss = P.lr;
            params.lr_self_right_win = P.lr;
            params.lr_self_right_loss = P.lr;
            single_lr = 1;
        elseif strcmp(fields{i},'lr_advie')
            params.lr_advice_left_win = P.lr_advice;
            params.lr_advice_left_loss = P.lr_advice;
            params.lr_advice_right_win = P.lr_advice;
            params.lr_advice_right_loss = P.lr_advice;
            single_lr_advice = 1;

        elseif strcmp(fields{i},'lr_self')
            params.lr_self_left_win = P.lr_self;
            params.lr_self_left_loss = P.lr_self;
            params.lr_self_right_win = P.lr_self;
            params.lr_self_right_loss = P.lr_self;
        elseif strcmp(fields{i},'lr_left')
            params.lr_advice_left_win = P.lr_left;
            params.lr_advice_left_loss = P.lr_left;
            params.lr_self_left_win = P.lr_left;
            params.lr_self_left_loss = P.lr_left;
            single_lr_left = 1;
        elseif strcmp(fields{i},'lr_right')
            params.lr_advice_right_win = P.lr_right;
            params.lr_advice_right_loss = P.lr_right;
            params.lr_self_right_win = P.lr_right;
            params.lr_self_right_loss = P.lr_right;
        elseif strcmp(fields{i},'lr_win')
            params.lr_advice_left_win = P.lr_win;
            params.lr_advice_right_win = P.lr_win;
            params.lr_self_left_win = P.lr_win;
            params.lr_self_right_win = P.lr_win;
            single_lr_win = 1;
        elseif strcmp(fields{i},'lr_loss')
            params.lr_advice_left_loss = P.lr_loss;
            params.lr_advice_right_loss = P.lr_loss;
            params.lr_self_left_loss = P.lr_loss;
            params.lr_self_right_loss = P.lr_loss;
        elseif strcmp(fields{i},'discount_factor')
            params.discount_factor = P.discount_factor;
        elseif strcmp(fields{i},'inv_temp')
            params.inv_temp = P.inv_temp;
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

        % if subject chose advice
        % Q(advise) = Q(advise) + lr(R_advise + discount_factor * max(Q(advise_next)) - Q(advise));
        if actual_actions(1) == 3
            % determine the which lr to use
            if actual_actions(2) == 1
                if actual_reward > 0
                    lr = params.lr_advice_left_win;
                else
                    lr = params.lr_advice_left_loss;
                end
            else
                if actual_reward > 0
                    lr = params.lr_advice_right_win;
                else
                    lr = params.lr_advice_right_loss;
                end
            end
            % update the Q table
            q_model.q_table(1,3) = q_model.q_table(1,3) + lr*(0 + params.discount_factor * max([q_model.q_table(1,1),q_model.q_table(1,2),q_model.q_table(1,3)]) - q_model.q_table(1,3));
        elseif actual_actions(1) == 1
        % if the subject chose left
            % determine the which lr to use
            if actual_reward > 0
                lr = params.lr_self_left_win;
            else
                lr = params.lr_self_left_loss;
            end
            % update the Q table
            q_model.q_table(1,1) = q_model.q_table(1,1) + lr*(actual_reward + params.discount_factor * max([q_model.q_table(1,1),q_model.q_table(1,2),q_model.q_table(1,3)]) - q_model.q_table(1,1));
        else
            % if the subject chose right
            % Q(right) = Q(right) + lr(R_right + discount_factor * max(Q(right_next)) - Q(right));
            % determine the which lr to use
            if actual_reward > 0
                lr = params.lr_self_right_win;
            else
                lr = params.lr_self_right_loss;
            end
            % update the Q table
            q_model.q_table(1,2) = q_model.q_table(1,2) + lr*(actual_reward + params.discount_factor * max([q_model.q_table(1,1), q_model.q_table(1,2), q_model.q_table(1,3)]) - q_model.q_table(1,2));
        end



        % for time step 2, check the len of the actual actions, countinue for those who have len > 2
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
                % if the subject chose left
                % Q(left) = Q(left) + lr(R_left + discount_factor * max(Q(left_next)) - Q(left));
                % determine the which lr to use
                if actual_reward > 0
                    lr = params.lr_advice_left_win;
                else
                    lr = params.lr_advice_left_loss;
                end
                % update the Q table
            
                q_model.q_table(after_advice_state,1) = q_model.q_table(after_advice_state,1) + lr*(actual_reward + params.discount_factor * max([q_model.q_table(after_advice_state,1),q_model.q_table(after_advice_state,2),q_model.q_table(after_advice_state,3)]) - q_model.q_table(after_advice_state,1));
            else    
                % if the subject chose right
                % Q(right) = Q(right) + lr(R_right + discount_factor * max(Q(right_next)) - Q(right));
                % determine the which lr to use
                if actual_reward > 0
                    lr = params.lr_advice_right_win;
                else
                    lr = params.lr_advice_right_loss;
                end
                % update the Q table
                q_model.q_table(after_advice_state,2) = q_model.q_table(after_advice_state,2) + lr*(actual_reward + params.discount_factor * max([q_model.q_table(after_advice_state,1), q_model.q_table(after_advice_state,2), q_model.q_table(after_advice_state,3)]) - q_model.q_table(after_advice_state,2));
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

