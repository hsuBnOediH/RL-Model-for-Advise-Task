function [results] = ModelFreeRLModelconnect_TT(task, MDP, params, sim)

% observations.hints = 0 is no hint, 1 is left hint, 2 is right hint
% observations.rewards(trial) 1 is win, 2 is loss
% choices : 1 is advisor, 2 is left, 3 is right
task.num_trials = 30;
task.num_blocks = 12;
observations.hints = nan(1,task.num_trials);
observations.rewards = nan(1,task.num_trials);
choices = nan(task.num_trials,2);

for trial=1:task.num_trials
    trial_info = MDP(trial);
    observations.hints(trial) = trial_info.o(1,2)-1;
    % if selected advisor
    if observations.hints(trial) 
        observations.rewards(trial) = 4 - trial_info.o(2,3); % ryan made win 1, loss 2
        choices(trial,1) = 1;
        choices(trial,2) = trial_info.o(3,3)-1; % left is 2, right 3
    else
        observations.rewards(trial) = 4 - trial_info.o(2,2); % ryan made win 1, loss 2
        choices(trial,1) = trial_info.o(3,2)-1; % left is 2, right 3
        choices(trial,2) = 0;
    end
    
end

params.p_right = .5;
single_omega = 0;
single_eta = 0;
%field = task.field;
field = fieldnames(params);
for i = 1:length(field)
    if strcmp(field{i},'omega')
        params.omega_d_win = params.omega;
        params.omega_d_loss = params.omega;
        params.omega_a_win = params.omega;
        params.omega_a_loss = params.omega;
        params.omega_d = params.omega;
        params.omega_a = params.omega;
        single_omega = 1;
    elseif strcmp(field{i},'eta')
        params.eta_d_win = params.eta;
        params.eta_d_loss = params.eta;
        params.eta_a_win = params.eta;
        params.eta_a_loss = params.eta;
        params.eta_d = params.eta;
        params.eta_a = params.eta;
        single_eta = 1;
    end
end

for i = 1:length(field)
    if strcmp(field{i},'omega_d') & single_omega ~= 1
        params.omega_d_win = params.omega_d;
        params.omega_d_loss = params.omega_d;
    elseif strcmp(field{i},'eta_d') & single_eta ~= 1
        params.eta_d_win = params.eta_d;
        params.eta_d_loss = params.eta_d;
    elseif strcmp(field{i},'omega_a') & single_omega ~= 1
        params.omega_a_win = params.omega_a;
        params.omega_a_loss = params.omega_a;
    elseif strcmp(field{i},'eta_a') & single_eta ~= 1
        params.eta_a_win = params.eta_a;
        params.eta_a_loss = params.eta_a;
    end
end




%RL model


% Assuming observations.rewards is a vector
actualreward = [MDP.actualreward]/10; % Copy the original vector and divided by 10



% Initialize matrices
action_probs = zeros(3, 2, trial);

qvalue = zeros(3, 3, trial);


if task.block_type == "SL"
     loss = 4;
elseif task.block_type == "LL"
     loss = params.l_loss_value;
end

%Initialization of q table (column: stage 1, advised left, advised left right; row: take advice, left, right)
qvalue(:, :, 1) = [(2*params.p_a-loss*(1-params.p_a))*params.reward_value, 0, 0;
                   (4*(1-params.p_right)-params.p_right*loss)*params.reward_value, (2*params.p_a-loss*(1-params.p_a))*params.reward_value, (2*(1-params.p_a)-loss*params.p_a)*params.reward_value;
                   (4*params.p_right-(1-params.p_right)*loss)*params.reward_value, (2*(1-params.p_a)-loss*params.p_a)*params.reward_value, (2*params.p_a-loss*(1-params.p_a))*params.reward_value];

for t = 1:trial

exp_values = exp(params.inv_temp * qvalue(:, 1, t));
action_probs(:, 1, t) = exp_values / sum(exp_values);

selected = choices(t, 1); % 1 (advice) or 2 (left) or 3 (right)

if actualreward(t) > 0

if selected == 2 || selected == 3
    % Update for the selected choice
    qvalue(selected, 1, t+1) = qvalue(selected, 1, t) + ...
        params.eta_d_win * (actualreward(t) * params.reward_value - qvalue(selected, 1, t));
    
    % Update for the opposite choice
    opposite = 5 - selected; % Maps 2 to 3 and 3 to 2
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t) + ...
        params.eta_d_win * (-loss * params.reward_value - qvalue(opposite, 1, t));

   % Forget qvalue(1, 1)
    qvalue(1, 1, t+1) = qvalue(1, 1, t) + ...
        params.omega_a_win * (qvalue(1, 1, 1) - qvalue(1, 1, t));

    % Forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t) + ...
        params.omega_a_win * (qvalue(2, 2, 1) - qvalue(2, 2, t));
    qvalue(3, 2, t+1) = qvalue(3, 2, t) + ...
        params.omega_a_win * (qvalue(3, 2, 1) - qvalue(3, 2, t));
    qvalue(2, 3, t+1) = qvalue(3, 2, t+1);
    qvalue(3, 3, t+1) = qvalue(2, 2, t+1);


elseif selected == 1  %advice
   hint = observations.hints(t)+1;

   exp_valuesafteradvice = exp(params.inv_temp * qvalue(2:3, hint, t));
   action_probs(2:3, 2, t) = exp_valuesafteradvice / sum(exp_valuesafteradvice);

   deltasecond = actualreward(t)*params.reward_value - qvalue(choices(t, 2), hint, t);
   deltafirst = qvalue(choices(t, 2), hint, t) - qvalue(1, 1, t);

   qvalue(1, 1, t+1) = qvalue(1, 1, t) + params.eta_a_win * deltafirst + params.eta_a_win * params.lamgda * deltasecond;
 
   secchoice = choices(t, 2);
          qvalue(secchoice, 1, t+1) = qvalue(secchoice, 1, t) + params.eta_d_win * (2*actualreward(t)*params.reward_value - qvalue(secchoice, 1, t));
          qvalue(secchoice, hint, t+1) = qvalue(secchoice, hint, t) + params.eta_a_win * (actualreward(t)*params.reward_value - qvalue(secchoice, hint, t));
          qvalue(5-secchoice, 1, t+1) = qvalue(5-secchoice, 1, t) + params.eta_d_win * (-loss*params.reward_value - qvalue(5-secchoice, 1, t));
          qvalue(5-secchoice, hint, t+1) = qvalue(5-secchoice, hint, t) + params.eta_a_win * (-loss*params.reward_value - qvalue(5-secchoice, hint, t));
          qvalue(2, 5-hint, t+1) = qvalue(3, hint, t+1);
          qvalue(3, 5-hint, t+1) = qvalue(2, hint, t+1);
end


elseif actualreward(t) < 0

if selected == 2 || selected == 3
    % Update for the selected choice
    qvalue(selected, 1, t+1) = qvalue(selected, 1, t) + ...
        params.eta_d_loss * (-loss * params.reward_value - qvalue(selected, 1, t));
    
    % Update for the opposite choice
    opposite = 5 - selected; % Maps 2 to 3 and 3 to 2
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t) + ...
        params.eta_d_loss * (4 * params.reward_value - qvalue(opposite, 1, t));
    

     % Forget qvalue(1, 1)
    qvalue(1, 1, t+1) = qvalue(1, 1, t) + ...
        params.omega_a_loss * (qvalue(1, 1, 1) - qvalue(1, 1, t));

    % Forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t) + ...
        params.omega_a_loss * (qvalue(2, 2, 1) - qvalue(2, 2, t));
    qvalue(3, 2, t+1) = qvalue(3, 2, t) + ...
        params.omega_a_loss * (qvalue(3, 2, 1) - qvalue(3, 2, t));
    qvalue(2, 3, t+1) = qvalue(3, 2, t+1);
    qvalue(3, 3, t+1) = qvalue(2, 2, t+1);


elseif selected == 1  %advice
   hint = observations.hints(t)+1;

   exp_valuesafteradvice = exp(params.inv_temp * qvalue(2:3, hint, t));
   action_probs(2:3, 2, t) = exp_valuesafteradvice / sum(exp_valuesafteradvice);

   deltasecond = -loss*params.reward_value - qvalue(choices(t, 2), hint, t);
   deltafirst = qvalue(choices(t, 2), hint, t) - qvalue(1, 1, t);

   qvalue(1, 1, t+1) = qvalue(1, 1, t) + params.eta_a_loss * deltafirst + params.eta_a_loss * params.lamgda * deltasecond;
 
   secchoice = choices(t, 2);
          qvalue(secchoice, 1, t+1) = qvalue(secchoice, 1, t) + params.eta_d_loss * (-loss*params.reward_value - qvalue(secchoice, 1, t));
          qvalue(secchoice, hint, t+1) = qvalue(secchoice, hint, t) + params.eta_a_loss * (-loss*params.reward_value - qvalue(secchoice, hint, t));
          qvalue(5-secchoice, 1, t+1) = qvalue(5-secchoice, 1, t) + params.eta_d_loss * (4*params.reward_value - qvalue(5-secchoice, 1, t));
          qvalue(5-secchoice, hint, t+1) = qvalue(5-secchoice, hint, t) + params.eta_a_loss * (2*params.reward_value - qvalue(5-secchoice, hint, t));
          qvalue(2, 5-hint, t+1) = qvalue(3, hint, t+1);
          qvalue(3, 5-hint, t+1) = qvalue(2, hint, t+1);


end

end

end


results.choices(:,:) = choices(:,:);

    results.input.task = task;
    results.input.params = params;
    results.input.observations = observations;
    results.input.choices = choices;
    results.input.sim = sim;

    results.blockwise.action_probs = action_probs;
    results.blockwise.actions = choices;




