function q_model = model_update(q_model, preprocessed_data)
    % Loop through each trial in the preprocessed data
    for i = 1:length(preprocessed_data)
        % Extract current state, action, reward, and next state from the data
        current_state = preprocessed_data(i).state;
        action = preprocessed_data(i).action;
        reward = preprocessed_data(i).reward;
        next_state = preprocessed_data(i).next_state;
        
        % Find the indices for the current state and action
        state_idx = find(strcmp(q_model.states, current_state));
        action_idx = find(strcmp(q_model.actions, action));
        
        % Find the index of the next state
        next_state_idx = find(strcmp(q_model.states, next_state));
        
        % Select appropriate learning rate based on the current state and action
        if strcmp(current_state, 'win')
            if strcmp(action, 'left')
                lr = q_model.lr_win_left;
            else
                lr = q_model.lr_win_right;
            end
        elseif strcmp(current_state, 'loss')
            if strcmp(action, 'left')
                lr = q_model.lr_loss_left;
            else
                lr = q_model.lr_loss_right;
            end
        elseif strcmp(current_state, 'advise_win')
            lr = q_model.lr_advise_win;
        elseif strcmp(current_state, 'advise_loss')
            lr = q_model.lr_advise_loss;
        else
            % Default learning rate if no specific condition matches
            lr = q_model.lr_advise_loss; % Adjust based on the model structure or default behavior
        end
        
        % Compute the maximum Q-value for the next state (for all possible actions)
        max_next_q = max(q_model.q_table(next_state_idx, :));
        
        % Calculate the Q-value update using the Q-learning formula
        q_model.q_table(state_idx, action_idx) = q_model.q_table(state_idx, action_idx) + ...
            lr * (reward + q_model.discount_factor * max_next_q - q_model.q_table(state_idx, action_idx));
    end
end