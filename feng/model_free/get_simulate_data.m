function  data = get_simulate_data(q_model, preprocessed_data)

    % Load trial_info from ../../trialinfo_forty_eighty.mat
    % col1, prob of advise is truth
    % col2, prob of left will win
    % col3, party size
    trial_info = load("../../trialinfo_forty_eighty.mat").trialinfo_forty_eighty;

    % Loop through each trial in the preprocessed data
    for i = 1:length(preprocessed_data)

        gt_prob_advise = str2double(trial_info{i, 1}); 
        gt_prob_left = str2double(trial_info{i, 2});  
        party_size = str2double(trial_info{i, 3});    
        
        % Calculate the probability of the right side winning
        gt_prob_right = 1 - gt_prob_left;
        
        % Sample the ground truth result based on probabilities
        is_left_winning = rand < gt_prob_left;
        
        % Sample the advisor's suggestion
        if rand < gt_prob_advise
            % Advisor is truthful, matches ground truth
            is_left_advise = is_left_winning;
        else
            % Advisor is lying, opposite of ground truth
            is_left_advise = ~is_left_winning;
        end
        
        % Initialize states and actions
        states = [];
        actions = [];

        start_state = 0;
        states(end+1) = start_state;
        first_potential_actions = [0, 1, 2];

        % Initialize q_values array
        first_q_values = nan(1, 3);
        for j = 1:3
            first_q_values(j) = q_model.q_table(start_state+1, first_potential_actions(j) + 1); % Adjust for MATLAB 1-based indexing
        end

        % Multiply by the inverse temperature and apply softmax
        temped_q_values = first_q_values * q_model.inv_temp;
        first_q_prob = softmax(temped_q_values);
        
        % Sample an action from the first_q_prob distribution
        first_action = randsample(first_potential_actions, 1, true, first_q_prob);
        actions(end+1) = first_action;

        if first_action == 2
            advise_state = 4;
            if is_left_advise
                advise_state = 3;
            end
            
            states(end+1) = advise_state;
            second_potential_actions = [0, 1];
            
            % Initialize second_q_values
            second_q_values = nan(1, 2);
            for j = 1:2
                second_q_values(j) = q_model.q_table(advise_state+1, second_potential_actions(j) + 1);
            end
            
            % Multiply by the inverse temperature and apply softmax
            temped_second_q_values = second_q_values * q_model.inv_temp;
            second_q_prob = softmax(temped_second_q_values);

            % Sample an action from second_q_prob
            second_action = randsample(second_potential_actions, 1, true, second_q_prob);
            actions(end+1) = second_action;
        end
        
        % Determine if advise was used
        with_advise = (length(actions) == 2);

        % Determine the final action and reward states based on ground truth
        final_action = actions(end);
        reward_states = 1; % Default reward state
        
        if final_action == is_left_winning && with_advise
            reward_states = 6;
        elseif final_action == is_left_winning && ~with_advise
            reward_states = 2;
        elseif final_action ~= is_left_winning && with_advise
            reward_states = 5;
        end

        % Update q_model.q_table or any other necessary structures here based on reward_states
        % For example:
        % q_model.q_table(states, actions) = updated_values; % Example placeholder logic
    end
end