%% Step by step introduction to building and using active inference models
function [gen_data] = advise_simTT(params, plot, model, OMEGAPOSINEGA)

% rng('shuffle') % This sets the random number generator to produce a different 
%                % random sequence each time, which leads to variability in 
%                % repeated simulation results (you can alse set to 'default'
%                % to produce the same random sequence each time)

% if (room == 60)
%     load('trialinfo_sixty.mat')
%     trialinfo = trialinfo_sixty;
% elseif (room == 100)
%     load('trialinfo_hundred.mat')
%     trialinfo = trialinfo_hundred;
% end           
load('trialinfo_forty_eighty.mat');
              
      

    trialinfo = trialinfo_forty_eighty;
    
    all_MDPs = [];
    for idx_block = 1:12
        for idx_trial = 1:30
            task.true_p_right(idx_trial) = 1-str2double(trialinfo{(idx_block-1)*30+idx_trial,2});
            task.true_p_a(idx_trial) = str2double(trialinfo{(idx_block-1)*30+idx_trial,1});
        end
        if strcmp(trialinfo{idx_block*30-29,3}, '80')
            task.block_type = "LL";
        else
            task.block_type = "SL";
        end
        MDP = [];
        sim = 1;
        
        if    model == 1
              MDPs  = Simple_Advice_Model_TT(task, MDP, params, sim);
        elseif model == 2
              MDPs  = ModelFreeRLModelconnect_TT(task, MDP, params, sim);
        elseif model == 3
            if OMEGAPOSINEGA
              MDPs  = ModelFreeRLModeldisconnectPosiNegaForget_TT(task, MDP, params, sim);
            else
              MDPs  = ModelFreeRLModeldisconnect_TT(task, MDP, params, sim);
            end
        end

        all_MDPs = [all_MDPs; MDPs'];
    end



% Initialize an empty array to store the merged data
merged_actions = [];

% Loop through each element in all_MDPs
for i = 1:numel(all_MDPs)
    % Extract the blockwise.actions field
    blockactions = all_MDPs(i).blockwise.actions;
    
    % Concatenate vertically
    merged_actions = [merged_actions; blockactions];
end

merged_hints = [];

% Loop through each element in all_MDPs
for i = 1:numel(all_MDPs)
    % Extract the blockwise.actions field
    blockhints = all_MDPs(i).blockwise.hint_outcomes;
    
    % Concatenate
    merged_hints = [merged_hints blockhints];
end

merged_rewards = [];

% Loop through each element in all_MDPs
for i = 1:numel(all_MDPs)
    % Extract the blockwise.actions field
    blockrewards = all_MDPs(i).blockwise.reward_outcomes;
    
    % Concatenate
    merged_rewards = [merged_rewards blockrewards];
end

merged_actualrewards = [];

% Loop through each element in all_MDPs
for i = 1:numel(all_MDPs)
    % Extract the blockwise.actions field
    blockactualrewards = all_MDPs(i).blockwise.actualreward;
    
    % Concatenate
    merged_actualrewards = [merged_actualrewards blockactualrewards];
end


for n = 1:length(merged_rewards)
        pt = 4 - merged_rewards(n);

        if merged_hints(n) > 0
            r = merged_actions(n,2) + 1;  

            u{n} = [1 2; 1 r]';
            
            y = merged_hints(n) + 1;

            o{n} = [1 y 1; 1 1 pt; 1 2 r];

        elseif merged_hints(n) == 0
            r = merged_actions(n,1) + 1;  

            u{n} = [1 r; 1 1]';
            o{n} = [1 1 1; 1 pt 1; 1 r 1];
        end

end

if plot
    N = numel(o); % Number of elements (360 in this case)

    % Initialize the struct array
    whole_MDP(N,1) = struct('o', [], 'u', []);

    % Loop through and populate the struct fields
    for i = 1:N
    whole_MDP(i).o = o{i};                     % Assign the i-th element of o
    whole_MDP(i).u = u{i};                     % Assign the i-th element of u
    end
    advise_plot_tt(whole_MDP);
end

gen_data = struct(                ...
    'observations', {o}',       ...
    'responses', {u}',           ...
    'actualrewards', {merged_actualrewards}',           ...
    'trialinfo', {trialinfo_forty_eighty}'  ...
);

%clear MDP
clear MDP trialinfo room