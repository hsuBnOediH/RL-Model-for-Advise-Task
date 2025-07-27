function [results] = ModelFreeRLModeldisconnect_TT(task, MDP, params, sim)

%%%Specify forget or non-forget model
FORGET = true;

% observations.hints = 0 is no hint, 1 is left hint, 2 is right hint
% observations.rewards(trial) 1 is win, 2 is loss
% choices : 1 is advisor, 2 is left, 3 is right
task.num_trials = 30;
task.num_blocks = 12;
observations.hints = nan(1,task.num_trials);
observations.rewards = nan(1,task.num_trials);
choices = nan(task.num_trials,2);

if sim == 0
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

p_win = 1;

hint_outcomes(1,1:task.num_trials) = 0;
actions = zeros(task.num_trials,2);

    for trial = 1:task.num_trials
        hint_outcome_vector(:,trial) = [0 0]';
        true_context(:,:,trial) = zeros(2,1);
        true_A{1}(:,:,trial) = zeros(2,2);
        A{2}(:,:,trial) = zeros(2,2);
        p_o_win(:,:,trial) = zeros(2,3);
        if sim == 1
            true_context_vector(trial) = find(rand < cumsum([1-task.true_p_right(trial) task.true_p_right(trial)]'),1)-1;
        end
    end

%left better
        A{2}(:,:,1) = [p_win   1-p_win; % win
                       1-p_win p_win];  % lose
        
%right better
        A{2}(:,:,2) = [1-p_win p_win;   % win
                       p_win   1-p_win];% lose

        
%RL model


% Assuming observations.rewards is a vector
if sim == 0
actualreward = [MDP.actualreward]/10; % Copy the original vector and divided by 10
end


% Initialize matrices
action_probs = zeros(3, 2, task.num_trials);

qvalue = zeros(3, 3, task.num_trials);


if task.block_type == "SL"
     loss = params.l_loss_value;
elseif task.block_type == "LL"
     loss = params.l_loss_value*params.Rsensitivity;
end

%Initialization of q table (column: start, advised left, advised right; row: take advice, left, right)
qvalue(:, :, 1) = [2*params.p_a-loss*(1-params.p_a),         0,                                0;
                   4*(1-params.p_right)-params.p_right*loss, 2*params.p_a-loss*(1-params.p_a), 2*(1-params.p_a)-loss*params.p_a;
                   4*params.p_right-(1-params.p_right)*loss, 2*(1-params.p_a)-loss*params.p_a, 2*params.p_a-loss*(1-params.p_a)];

qvalue(:, :, 1) = qvalue(:, :, 1)*params.reward_value;

for t = 1:task.num_trials

%exp_values = exp(params.inv_temp * qvalue(:, 1, t));

%exp_values = exp(params.inv_temp * (qvalue(:, 1, t) - max(qvalue(:, 1, t))));
%action_probs(:, 1, t) = exp_values / sum(exp_values);

 action_probs(:, 1, t) = spm_softmax(params.inv_temp*qvalue(:,1,t))';

  if sim == 1
     true_context(:,:,t) = [1-true_context_vector(t) true_context_vector(t)]';
     true_A{1}(:,:,t) =  [task.true_p_a(t)   1-task.true_p_a(t); % "try left"
                          1-task.true_p_a(t) task.true_p_a(t)];  % "try right"
     true_p_o_hint(:,t) = true_A{1}(:,:,1)*true_context(:,:,t);
     true_p_o_win(:,2,t) = A{2}(:,:,1)*true_context(:,:,t);
     true_p_o_win(:,3,t) = A{2}(:,:,2)*true_context(:,:,t);

     actions(t, 1) = find(rand < cumsum(action_probs(:, 1, t)'), 1);

  elseif sim == 0
     actions(t, 1) = choices(t, 1);

  end

 selected = actions(t, 1); % 1 (ask advice) or 2 (left) or 3 (right)

 

 if selected ~= 1

       if sim == 1
                hint_outcomes(t) = 0;

                reward_outcomes(t) = find(rand < cumsum(true_p_o_win(:,actions(t,1),t)'),1);
                % make actualrewards for simulation
                if reward_outcomes(t) == 1
                   actualreward(t) = 4;
                elseif reward_outcomes(t) == 2
                  if task.block_type == "SL"
                    actualreward(t) = -4;
                  elseif task.block_type == "LL"
                    actualreward(t) = -8;
                  end
                end
       else
           reward_outcomes(t) = observations.rewards(t);
       end


    if actualreward(t) > 0

    % Update for the selected choice
    qvalue(selected, 1, t+1) = qvalue(selected, 1, t) + ...
        params.eta_d_win * (actualreward(t) * params.reward_value - qvalue(selected, 1, t));

    opposite = 5 - selected; % If selected is 2, opposite is 3; if selected is 3, opposite is 2

     if FORGET

%     % Forget the opposite choice
%     qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t) + ...
%         params.omega_d_win * (qvalue(opposite, 1, 1) - qvalue(opposite, 1, t));
    
    % No forget the opposite choice
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t);
   
    % Forget qvalue(1, 1)
    qvalue(1, 1, t+1) = qvalue(1, 1, t) + ...
        params.omega_a_win * (qvalue(1, 1, 1) - qvalue(1, 1, t));

    % Forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t) + ...
        params.omega_a_win * (qvalue(2, 2, 1) - qvalue(2, 2, t));
    qvalue(3, 2, t+1) = qvalue(3, 2, t) + ...
        params.omega_a_win * (qvalue(3, 2, 1) - qvalue(3, 2, t));

     else

    % No forget the opposite choice
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t);

    % No forget advice
    qvalue(1, 1, t+1) = qvalue(1, 1, t);

    % No forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t);
    qvalue(3, 2, t+1) = qvalue(3, 2, t);
    
     end

    qvalue(2, 3, t+1) = qvalue(3, 2, t+1);
    qvalue(3, 3, t+1) = qvalue(2, 2, t+1);


    elseif actualreward(t) < 0

    % Update for the selected choice
    qvalue(selected, 1, t+1) = qvalue(selected, 1, t) + ...
        params.eta_d_loss * (-loss * params.reward_value - qvalue(selected, 1, t));
    
    % Forget the opposite choice
    opposite = 5 - selected; % If selected is 2, opposite is 3; if selected is 3, opposite is 2

     if FORGET
%     % Forget the opposite choice
%     qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t) + ...
%         params.omega_d_loss * (qvalue(opposite, 1, 1) - qvalue(opposite, 1, t));

    % No forget the opposite choice
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t);

    % Forget qvalue(1, 1)
    qvalue(1, 1, t+1) = qvalue(1, 1, t) + ...
        params.omega_a_loss * (qvalue(1, 1, 1) - qvalue(1, 1, t));

    % Forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t) + ...
        params.omega_a_loss * (qvalue(2, 2, 1) - qvalue(2, 2, t));
    qvalue(3, 2, t+1) = qvalue(3, 2, t) + ...
        params.omega_a_loss * (qvalue(3, 2, 1) - qvalue(3, 2, t));

     else
    % No forget the opposite choice
    qvalue(opposite, 1, t+1) = qvalue(opposite, 1, t);

    % No forget advice
    qvalue(1, 1, t+1) = qvalue(1, 1, t);

    % No forget choices after advice
    qvalue(2, 2, t+1) = qvalue(2, 2, t);
    qvalue(3, 2, t+1) = qvalue(3, 2, t);
    
     end
    
    qvalue(2, 3, t+1) = qvalue(3, 2, t+1);
    qvalue(3, 3, t+1) = qvalue(2, 2, t+1);


    end


 elseif selected == 1  %advice

   if sim == 1
                hint_outcomes(t) = find(rand < cumsum(true_p_o_hint(:,t)'),1);
   else 
                hint_outcomes(t) = observations.hints(t);
   end

   hint = hint_outcomes(t)+1;

%   exp_valuesafteradvice = exp(params.inv_temp * qvalue(2:3, hint, t));

%   exp_valuesafteradvice = exp(params.inv_temp * (qvalue(2:3, hint, t) - max(qvalue(2:3, hint, t))));
%   action_probs(2:3, 2, t) = exp_valuesafteradvice / sum(exp_valuesafteradvice);

    action_probs(:,2,t) = [0; spm_softmax(params.inv_temp*qvalue(2:3,hint,t))]';

    % select actions
    if sim == 1
           actions(t,2) = find(rand < cumsum(action_probs(:,2,t)'),1);
    else
           actions(t,2) = choices(t,2);
    end

    if sim == 1
           reward_outcomes(t) = find(rand < cumsum(true_p_o_win(:,actions(t,2),t)'),1);
           if reward_outcomes(t) == 1
                actualreward(t) = 2;
           elseif reward_outcomes(t) == 2
                 if task.block_type == "SL"
                    actualreward(t) = -4;
                 elseif task.block_type == "LL"
                    actualreward(t) = -8;
                 end
           end   
    else
        reward_outcomes(t) = observations.rewards(t);
    end    

   if actualreward(t) > 0

   deltasecond = actualreward(t)*params.reward_value - qvalue(actions(t, 2), hint, t);
   deltafirst = qvalue(actions(t, 2), hint, t) - qvalue(1, 1, t);

   qvalue(1, 1, t+1) = qvalue(1, 1, t) + params.eta_a_win * deltafirst + params.eta_a_win * params.lamgda * deltasecond;
 
   secchoice = actions(t, 2);
          qvalue(secchoice, 1, t+1) = qvalue(secchoice, 1, t) + params.eta_d_win * (2*actualreward(t)*params.reward_value - qvalue(secchoice, 1, t));
          qvalue(secchoice, hint, t+1) = qvalue(secchoice, hint, t) + params.eta_a_win * (actualreward(t)*params.reward_value - qvalue(secchoice, hint, t));
          
          
          secopposite = 5 - secchoice;
          
%            if FORGET
%           % Forget the opposite choice
%           qvalue(secopposite, 1, t+1) = qvalue(secopposite, 1, t) + params.omega_d_win * (qvalue(secopposite, 1, 1) - qvalue(secopposite, 1, t));
%           qvalue(secopposite, hint, t+1) = qvalue(secopposite, hint, t) + params.omega_a_win * (qvalue(secopposite, hint, 1) - qvalue(secopposite, hint, t));
% 
%            else

          % No forget the opposite choice
          qvalue(secopposite, 1, t+1) = qvalue(secopposite, 1, t);
          qvalue(secopposite, hint, t+1) = qvalue(secopposite, hint, t);

%            end
          
          %Same updates as the advised option
          qvalue(2, 5-hint, t+1) = qvalue(3, hint, t+1);
          qvalue(3, 5-hint, t+1) = qvalue(2, hint, t+1);
          
          
   elseif actualreward(t) < 0
   
   deltasecond = -loss*params.reward_value - qvalue(actions(t, 2), hint, t);
   deltafirst = qvalue(actions(t, 2), hint, t) - qvalue(1, 1, t);

   qvalue(1, 1, t+1) = qvalue(1, 1, t) + params.eta_a_loss * deltafirst + params.eta_a_loss * params.lamgda * deltasecond;
 
   secchoice = actions(t, 2);
          qvalue(secchoice, 1, t+1) = qvalue(secchoice, 1, t) + params.eta_d_loss * (-loss*params.reward_value - qvalue(secchoice, 1, t));
          qvalue(secchoice, hint, t+1) = qvalue(secchoice, hint, t) + params.eta_a_loss * (-loss*params.reward_value - qvalue(secchoice, hint, t));

          
          secopposite = 5 - secchoice;

%           if FORGET
%           % Forget the opposite choice
%           qvalue(secopposite, 1, t+1) = qvalue(secopposite, 1, t) + params.omega_d_loss * (qvalue(secopposite, 1, 1) - qvalue(secopposite, 1, t));
%           qvalue(secopposite, hint, t+1) = qvalue(secopposite, hint, t) + params.omega_a_loss * (qvalue(secopposite, hint, 1) - qvalue(secopposite, hint, t));
% 
%           else

          % No forget the opposite choice
          qvalue(secopposite, 1, t+1) = qvalue(secopposite, 1, t);
          qvalue(secopposite, hint, t+1) = qvalue(secopposite, hint, t);

%           end
          
          %Same updates as the advised option
          qvalue(2, 5-hint, t+1) = qvalue(3, hint, t+1);
          qvalue(3, 5-hint, t+1) = qvalue(2, hint, t+1);


    end

 end
 
end


results.choices(:,:) = actions(:,:);

    results.input.task = task;
    results.input.params = params;
    results.input.observations = observations;
    results.input.choices = choices;
    results.input.sim = sim;

    results.blockwise.action_probs = action_probs;
    results.blockwise.actions = actions;
    results.blockwise.hint_outcomes = hint_outcomes;
    results.blockwise.reward_outcomes = reward_outcomes;
    results.blockwise.actualreward = actualreward*10;


