function [results] = ModelBasedRLModeldisconnectPosiNegaForget_TT(task, MDP, params, sim)

%%% Specify
FORGETopposite = true;
FORGETtoZero = true;

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

single_omega = 0;
single_eta = 0;
field = fieldnames(params);

for i = 1:length(field)
    if strcmp(field{i,1}, 'omega')
      single_omega = 1;
      params.omega_d_posi = params.omega;
      params.omega_d_nega = params.omega;
      params.omega_a_posi = params.omega;
      params.omega_a_nega = params.omega;
    end
    if strcmp(field{i,1}, 'eta')
      single_eta = 1;
      params.eta_d_win = params.eta;
      params.eta_d_loss = params.eta;
      params.eta_a_win = params.eta;
      params.eta_a_loss = params.eta;
      params.lamgda = 0; % (kept for compatibility; not used in MB planning)
    end
end

if ~single_eta
    if ismember('eta_d', field)
      params.eta_d_win = params.eta_d;
      params.eta_d_loss = params.eta_d;
    end
    if ismember('eta_a', field)
      params.eta_a_win = params.eta_a;
      params.eta_a_loss = params.eta_a;
    end
end

% Initialize matrices
action_probs = zeros(3, 2, task.num_trials);
qvalue = zeros(3, 3, task.num_trials);

if task.block_type == "SL"
     loss = params.l_loss_value;
elseif task.block_type == "LL"
     loss = params.l_loss_value*params.Rsensitivity;
end

% ---- Keep same qvalue initialization ----
qvalue(:, :, 1) = [(params.reward_value/2)*params.p_a-loss*(1-params.p_a),      0,                                                      0;
                   params.reward_value*.5-loss*.5,   (params.reward_value/2)*params.p_a-loss*(1-params.p_a), (params.reward_value/2)*(1-params.p_a)-loss*params.p_a;
                   params.reward_value*.5-loss*.5,   (params.reward_value/2)*(1-params.p_a)-loss*params.p_a, (params.reward_value/2)*params.p_a-loss*(1-params.p_a)];

% =========================
% Model-based: learn environment model and plan
% =========================
% Reward model: Rhat(action, state) in *value units of qvalue*
% action: 2=left, 3=right ; state: 1=start, 2=advised-left, 3=advised-right
Rhat = zeros(3,3);
Rhat(2:3,1:3) = qvalue(2:3,1:3,1); % initialize from MF initial values

% Advice hint transition model when asking advice: P(hint=left/right)
% store as [P(leftHint) P(rightHint)], initialize symmetric
Phint = [0.5 0.5];

% For simulation bookkeeping (same outputs)
actions = zeros(task.num_trials,2);
hint_outcomes = zeros(1,task.num_trials);
reward_outcomes = zeros(1,task.num_trials);
actualreward = zeros(1,task.num_trials);

% true_* are only used in sim==1 branch below (kept from MF style)
true_context = zeros(2,1,task.num_trials);
true_A = cell(1,2);
true_p_o_hint = zeros(2,task.num_trials);
true_p_o_win = zeros(2,3,task.num_trials);

for t = 1:task.num_trials

  % ---- 1st-stage choice probabilities (same) ----
  action_probs(:, 1, t) = spm_softmax(params.inv_temp*qvalue(:,1,t))';

  % select first-stage action
  if sim == 1
     % (kept, but this file is mainly for fit; simulation can be extended if needed)
     actions(t, 1) = find(rand < cumsum(action_probs(:, 1, t)'), 1);
  else
     actions(t, 1) = choices(t, 1);
  end

  selected = actions(t, 1); % 1 (ask advice) or 2 (left) or 3 (right)

  if selected ~= 1
    % =========================
    % Direct choice (no advice)
    % =========================
    if sim == 1
        hint_outcomes(t) = 0;
        % NOTE: sim branch kept minimal; actual reward generation not expanded here
    else
        reward_outcomes(t) = observations.rewards(t);
    end

    % Map reward_outcomes into actualreward (keep MF mapping)
    if sim == 0
        if reward_outcomes(t) == 1
            actualreward(t) = 4;
        elseif reward_outcomes(t) == 2
            if task.block_type == "SL"
                actualreward(t) = -4;
            elseif task.block_type == "LL"
                actualreward(t) = -8;
            end
        end
    end

    % --- Model learning: update reward model Rhat(selected, state=1) ---
    if actualreward(t) > 0
        target = (params.reward_value + params.self_reliance_bonus);
        Rhat(selected,1) = Rhat(selected,1) + params.eta_d_win * (target - Rhat(selected,1));
    elseif actualreward(t) < 0
        target = (-loss + params.self_reliance_bonus);
        Rhat(selected,1) = Rhat(selected,1) + params.eta_d_loss * (target - Rhat(selected,1));
    end

    % --- Forget opposite (same policy as MF, but apply to Rhat) ---
    opposite = 5 - selected; % 2<->3
    if FORGETopposite
      if FORGETtoZero
        if 0 <= Rhat(opposite,1)
          Rhat(opposite,1) = Rhat(opposite,1) + params.omega_d_posi * (0 - Rhat(opposite,1));
        else
          Rhat(opposite,1) = Rhat(opposite,1) + params.omega_d_nega * (0 - Rhat(opposite,1));
        end
      else
        % toward initial value
        if qvalue(opposite,1,1) <= Rhat(opposite,1)
          Rhat(opposite,1) = Rhat(opposite,1) + params.omega_d_posi * (qvalue(opposite,1,1) - Rhat(opposite,1));
        else
          Rhat(opposite,1) = Rhat(opposite,1) + params.omega_d_nega * (qvalue(opposite,1,1) - Rhat(opposite,1));
        end
      end
    end

    % --- Planning: rebuild qvalue for next trial ---
    qvalue(:,:,t+1) = qvalue(:,:,t); % default carry
    qvalue(2,1,t+1) = Rhat(2,1);
    qvalue(3,1,t+1) = Rhat(3,1);
    qvalue(2,2,t+1) = Rhat(2,2);
    qvalue(3,2,t+1) = Rhat(3,2);
    qvalue(2,3,t+1) = Rhat(2,3);
    qvalue(3,3,t+1) = Rhat(3,3);

    % Value of asking advice at start = E_hint[ max_a Q(hint,a) ]
    v2 = max([qvalue(2,2,t+1), qvalue(3,2,t+1)]);
    v3 = max([qvalue(2,3,t+1), qvalue(3,3,t+1)]);
    qvalue(1,1,t+1) = Phint(1)*v2 + Phint(2)*v3;

  else
    % =========================
    % Ask advice (2-stage)
    % =========================
    if sim == 1
        % kept minimal
    else
        hint_outcomes(t) = observations.hints(t);
    end
    hint = hint_outcomes(t)+1; % 2=left hint, 3=right hint (1=start)

    % 2nd-stage choice probabilities (same)
    action_probs(:,2,t) = [0; spm_softmax(params.inv_temp*qvalue(2:3,hint,t))]';

    if sim == 1
        actions(t,2) = find(rand < cumsum(action_probs(:,2,t)'),1);
    else
        actions(t,2) = choices(t,2);
    end

    % observe reward
    if sim == 0
        reward_outcomes(t) = observations.rewards(t);
        if reward_outcomes(t) == 1
            actualreward(t) = 2; % MF: advice branch win -> 2 (then *10 at output)
        elseif reward_outcomes(t) == 2
            if task.block_type == "SL"
                actualreward(t) = -4;
            elseif task.block_type == "LL"
                actualreward(t) = -8;
            end
        end
    end

    % % --- Model learning: update hint transition model Phint when advice is taken ---
    % if hint == 2 || hint == 3
    %     % use params.eta (if exists) else fallback small
    %     if ismember('eta', field)
    %         lrT = params.eta;
    %     else
    %         lrT = 0.1;
    %     end
    %     targetT = [double(hint==2) double(hint==3)];
    %     Phint = (1-lrT)*Phint + lrT*targetT;
    %     Phint = Phint ./ sum(Phint); % safety normalize
    % end

    secchoice = actions(t,2); % 2 or 3
    secopposite = 5 - secchoice;

    % --- Update reward model for chosen action ---
    if actualreward(t) > 0
        % direct-state(1) update uses full reward_value
        
        Rhat(secchoice,1) = Rhat(secchoice,1) + params.eta_d_win * (params.reward_value - Rhat(secchoice,1));

        % hint-state update uses reward_value/2
        
        Rhat(secchoice,hint) = Rhat(secchoice,hint) + params.eta_a_win * ((params.reward_value/2) - Rhat(secchoice,hint));

    elseif actualreward(t) < 0
        
        Rhat(secchoice,1) = Rhat(secchoice,1) + params.eta_d_loss * (-loss - Rhat(secchoice,1));

        Rhat(secchoice,hint) = Rhat(secchoice,hint) + params.eta_a_loss * (-loss - Rhat(secchoice,hint));
    end

    % --- Forget opposite action in both state=1 and hint-state ---
    if FORGETopposite
      if FORGETtoZero
        % state 1
        if 0 <= Rhat(secopposite,1)
          Rhat(secopposite,1) = Rhat(secopposite,1) + params.omega_d_posi * (0 - Rhat(secopposite,1));
        else
          Rhat(secopposite,1) = Rhat(secopposite,1) + params.omega_d_nega * (0 - Rhat(secopposite,1));
        end
        % hint state
        if 0 <= Rhat(secopposite,hint)
          Rhat(secopposite,hint) = Rhat(secopposite,hint) + params.omega_a_posi * (0 - Rhat(secopposite,hint));
        else
          Rhat(secopposite,hint) = Rhat(secopposite,hint) + params.omega_a_nega * (0 - Rhat(secopposite,hint));
        end
      else
        % toward initial values
        if qvalue(secopposite,1,1) <= Rhat(secopposite,1)
          Rhat(secopposite,1) = Rhat(secopposite,1) + params.omega_d_posi * (qvalue(secopposite,1,1) - Rhat(secopposite,1));
        else
          Rhat(secopposite,1) = Rhat(secopposite,1) + params.omega_d_nega * (qvalue(secopposite,1,1) - Rhat(secopposite,1));
        end

        if qvalue(secopposite,hint,1) <= Rhat(secopposite,hint)
          Rhat(secopposite,hint) = Rhat(secopposite,hint) + params.omega_a_posi * (qvalue(secopposite,hint,1) - Rhat(secopposite,hint));
        else
          Rhat(secopposite,hint) = Rhat(secopposite,hint) + params.omega_a_nega * (qvalue(secopposite,hint,1) - Rhat(secopposite,hint));
        end
      end
    end

    % --- Planning: rebuild qvalue for next trial ---
    qvalue(:,:,t+1) = qvalue(:,:,t);
    qvalue(2,1,t+1) = Rhat(2,1);
    qvalue(3,1,t+1) = Rhat(3,1);
    qvalue(2,2,t+1) = Rhat(2,2);
    qvalue(3,2,t+1) = Rhat(3,2);
    qvalue(2,3,t+1) = Rhat(2,3);
    qvalue(3,3,t+1) = Rhat(3,3);

    v2 = max([qvalue(2,2,t+1), qvalue(3,2,t+1)]);
    v3 = max([qvalue(2,3,t+1), qvalue(3,3,t+1)]);
    qvalue(1,1,t+1) = Phint(1)*v2 + Phint(2)*v3;

    % keep symmetry lines
    qvalue(2, 5-hint, t+1) = qvalue(3, hint, t+1);
    qvalue(3, 5-hint, t+1) = qvalue(2, hint, t+1);

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

end