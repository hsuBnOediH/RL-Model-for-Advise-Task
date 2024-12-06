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
        reward = cellfun(@str2double, reward, 'UniformOutput', false);

        advice_flags = subdat.event_type == 9 & (strcmp(subdat.result, "try left") | strcmp(subdat.result, "try right"));
        advice_idxs = subdat.trial(advice_flags) + 1;
        advices = subdat.result(advice_flags);


        party_sizes = reward_table.trial_type;
        % party_sizes include string like '0.4_0.6_0.9_40', take the last number as the party size, split by '_' and take the last one
        
        % Loop through each string in praty_sizes
        for i = 1:length(party_sizes)
            % Split the string by '_' and take the last part
            split_parts = split(party_sizes{i}, '_');
            % Convert the last part to a number and store it in party_sizes
            party_sizes{i} = split_parts{end};
        end
        % Convert party_sizes to a double array
        party_sizes = cellfun(@str2double, party_sizes);

        


        % Initialize 'res' as an empty structure array with the same length as 'response'
        res(length(response)) = struct('states', [], 'actions', []);

        % Define mapping for states and actions
        state_map = struct('start', 1,'left',2,'right',3, 'lose', 4,'win', 5,  'advise_lose', 6, 'advise_win', 7);
        action_map = struct('left',1, 'right', 2, 'advise', 3);

        % Loop through each trial
        for i = 1:length(response)
            % Initialize states and actions for each trial
            states = [state_map.start];
            actions = [];
            rewards = [];
            party_size = party_sizes(i);
            
            % Check if the trial took advice
            if ~ismember(i, advice_idxs)
                % No advice taken, so use response only
                actions(end+1) = action_map.(lower(response{i}));
                
                % Check reward to set the win/lose state
                if reward{i} > 0
                    states(end+1) = state_map.win;
                    rewards(end+1) = 40;
                else
                    states(end+1) = state_map.lose;
                    % check the dinner size to see if it's large or small, large is -80, small is -40
                    if party_size == 80
                        rewards(end+1) = -80;
                    else
                        rewards(end+1) = -40;
                    end
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
                    rewards(end+1) = 20;
                else
                    states(end+1) = state_map.advise_lose;
                    % check the dinner size to see if it's large or small, large is -80, small is -40
                    if party_size == 80
                        rewards(end+1) = -80;
                    else
                        rewards(end+1) = -40;
                    end
                end
            end
            
            % Store the results in 'res' for the current trial
            res(i).states = states;
            res(i).actions = actions;
            res(i).rewards = rewards;
            res(i).party_size = party_size;
        end

    end

end