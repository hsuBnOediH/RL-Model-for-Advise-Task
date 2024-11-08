function res = get_preprocessed_data(subject, input_folder_path)

    % Initialize has_practice_effects to false
    has_practice_effects = false;
    % Manipulate Data
    directory = dir(input_folder_path);
    % Sort by date
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    [~, sorted_indices] = sort(dates);
    sorted_directory = directory(sorted_indices);

    index_array = find(arrayfun(@(n) contains(sorted_directory(n).name, ['active_trust_' subject]), 1:numel(sorted_directory)));
    if length(index_array) > 1
        disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
    end

    file = '';
    for k = 1:length(index_array)
        file_index = index_array(k);
        file = [input_folder_path '/' sorted_directory(file_index).name];

        subdat = readtable(file);
        if strcmp(class(subdat.trial), 'cell')
            subdat.trial = str2double(subdat.trial);
        end

        if any(cellfun(@(x) isequal(x, 'MAIN'), subdat.trial_type)) && (max(subdat.trial) ~= 359)
            has_practice_effects = true;
        end

        if max(subdat.trial) ~= 359
            continue;
        end

        subdat = subdat(max(find(ismember(subdat.trial_type, 'MAIN'))) + 1:end, :);
        compressed = subdat(subdat.event_type == 4, :);
        
        response_table = subdat(subdat.event_type == 8, :);
        [~, idx] = unique(response_table.trial, 'first');
        response_table = response_table(idx, :);
        response = response_table.response;

        reward_table = subdat(subdat.event_type == 9 & ~(strcmp(subdat.result, "try left") | strcmp(subdat.result, "try right")), :);
        reward = reward_table.result;

        advice_flags = subdat.event_type == 9 & (strcmp(subdat.result, "try left") | strcmp(subdat.result, "try right"));
        advice_idxs = subdat.trial(advice_flags) + 1;
        advices = subdat.result(advice_flags);

        % Initialize 'res' as an empty structure array with the same length as 'response'
        res(length(response)) = struct('states', [], 'actions', []);

        % Define mapping for states and actions
        state_map = struct('start', 0, 'lose', 1, 'win', 2, 'left',3,'right',4, 'advise_lose', 5, 'advise_win', 6);
        action_map = struct('left', 0, 'right', 1, 'advise', 2);

        % Loop through each trial
        for i = 1:length(response)
            % Initialize states and actions for each trial
            states = [state_map.start];
            actions = [];
            
            % Check if the trial took advice
            if ~ismember(i, advice_idxs)
                % No advice taken, so use response only
                actions(end+1) = action_map.(lower(response{i}));
                
                % Check reward to set the win/lose state
                if reward{i} > 0
                    states(end+1) = state_map.win;
                else
                    states(end+1) = state_map.lose;
                end
            else
                % Advice was taken
                actions(end+1) = action_map.advise;  % Add advice action first
                actions(end+1) = action_map.(lower(response{i}));  % Then add the actual response

                idx = find(advice_idxs == i);
                states(end+1) = state_map.(strrep(advices{idx}, 'try ', ''));

                % Check reward to set the advised win/lose state
                if reward{i} > 0
                    states(end+1) = state_map.advise_win;
                else
                    states(end+1) = state_map.advise_lose;
                end
            end
            
            % Store the results in 'res' for the current trial
            res(i).states = states;
            res(i).actions = actions;
        end

    end

end