function [results] = ModelBasedRLModeldisconnectPosiNegaForget_TT(task, MDP, params, sim)

%%% Specify
FORGETopposite = true;
FORGETtoZero = true;
UPDATE_START_DIRECT_CHOICE = true;

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
    % If selected advisor
    if observations.hints(trial)
        observations.rewards(trial) = 4 - trial_info.o(2,3); % win=1, loss=2
        choices(trial,1) = 1;
        choices(trial,2) = trial_info.o(3,3)-1; % left=2, right=3 -> store as 2/3
    else
        observations.rewards(trial) = 4 - trial_info.o(2,2); % win=1, loss=2
        choices(trial,1) = trial_info.o(3,2)-1; % left=2, right=3 -> store as 2/3
        choices(trial,2) = 0;
    end
  end
end

% ----- parameter parsing (same style as MF) -----
single_omega = 0;
single_eta = 0;
field = fieldnames(params);
for i = 1:length(field)
    if strcmp(field{i},'omega')
        params.omega_d_posi = params.omega;
        params.omega_d_nega = params.omega;
        params.omega_a_posi = params.omega;
        params.omega_a_nega = params.omega;
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
    if strcmp(field{i},'omega_d') && single_omega ~= 1
        params.omega_d_posi = params.omega_d;
        params.omega_d_nega = params.omega_d;
    elseif strcmp(field{i},'eta_d') && single_eta ~= 1
        params.eta_d_win = params.eta_d;
        params.eta_d_loss = params.eta_d;
    elseif strcmp(field{i},'omega_a') && single_omega ~= 1
        params.omega_a_posi = params.omega_a;
        params.omega_a_nega = params.omega_a;
    elseif strcmp(field{i},'eta_a') && single_eta ~= 1
        params.eta_a_win = params.eta_a;
        params.eta_a_loss = params.eta_a;
    end
end

% ---- simulation bookkeeping (keep MF-compatible outputs) ----
hint_outcomes(1,1:task.num_trials) = 0;
actions = zeros(task.num_trials,2);

p_win = 1;
for trial = 1:task.num_trials
    true_context(:,:,trial) = zeros(2,1);
    true_A{1}(:,:,trial) = zeros(2,2);
    A{2}(:,:,trial) = zeros(2,2);
    p_o_win(:,:,trial) = zeros(2,3);
    if sim == 1
        true_context_vector(trial) = find(rand < cumsum([1-task.true_p_right(trial) task.true_p_right(trial)]'),1)-1;
    end
end
A{2}(:,:,1) = [p_win   1-p_win; 1-p_win p_win];
A{2}(:,:,2) = [1-p_win p_win;   p_win   1-p_win];

if sim == 0
    actualreward = [MDP.actualreward]/10;
end

action_probs = zeros(3, 2, task.num_trials);
qvalue = zeros(3, 3, task.num_trials);

if task.block_type == "SL"
     loss = params.l_loss_value;
elseif task.block_type == "LL"
     loss = params.l_loss_value*params.Rsensitivity;
end

% ===== State-PE internal beliefs: pW(action,state)=P(win|a,s) =====
% action index: 2=left, 3=right
% state index : 1=start, 2=LH, 3=RH
pW = 0.5*ones(3,3);              % neutral priors in probability space
Phint = [0.5 0.5];               % P(LH), P(RH)

% ----- IMPORTANT CHANGE: params.p_a as PRIOR TRUST (hint accuracy prior) -----
% Interpret params.p_a as "baseline trust before learning":
%   In LH (state=2): following the hint means choosing Left (action=2) is more likely to win.
%   In RH (state=3): following the hint means choosing Right (action=3) is more likely to win.
pa = clip01(params.p_a);
pW(2,2) = pa;        % P(win | choose Left, LH)  (follow)
pW(3,2) = 1 - pa;    % P(win | choose Right, LH) (oppose)
% Enforce the same symmetry convention as the MF code:
pW(2,3) = pW(3,2);   % P(win | choose Left, RH)
pW(3,3) = pW(2,2);   % P(win | choose Right, RH)

% Save the initial probabilities for FORGETtoZero==false case
pW_init = pW;

% Payoff mapping for expected value (EV)
Vwin_start = params.reward_value;
Vloss_start = -loss;
Vwin_hint  = (params.reward_value/2);
Vloss_hint = -loss;

% Build initial Q-values from pW (model-based EV computation)
qvalue(:,:,1) = build_q_from_p(pW, Phint, Vwin_start, Vloss_start, Vwin_hint, Vloss_hint);


for t = 1:task.num_trials

    % ---- 1st-stage choice probabilities ----
    % IMPORTANT CHANGE: no advice_bias_logit. Advice choice emerges from Q-values.
    logits = params.inv_temp*qvalue(:,1,t);
    action_probs(:, 1, t) = spm_softmax(logits)';

    if sim == 1
        true_context(:,:,t) = [1-true_context_vector(t) true_context_vector(t)]';
        true_A{1}(:,:,t) =  [task.true_p_a(t)   1-task.true_p_a(t);
                             1-task.true_p_a(t) task.true_p_a(t)];
        true_p_o_hint(:,t) = true_A{1}(:,:,1)*true_context(:,:,t);
        true_p_o_win(:,2,t) = A{2}(:,:,1)*true_context(:,:,t);
        true_p_o_win(:,3,t) = A{2}(:,:,2)*true_context(:,:,t);
        actions(t, 1) = find(rand < cumsum(action_probs(:, 1, t)'), 1);
    else
        actions(t, 1) = choices(t, 1);
    end

    selected = actions(t, 1); % 1 advice, 2 left, 3 right

    if selected ~= 1
        % =========================
        % Direct choice (no advice)
        % =========================
        hint_outcomes(t) = 0;

        if sim == 1
            reward_outcomes(t) = find(rand < cumsum(true_p_o_win(:,actions(t,1),t)'),1);
            if reward_outcomes(t) == 1
               actualreward(t) = 4;
            else
               actualreward(t) = (task.block_type=="SL")*(-4) + (task.block_type=="LL")*(-8);
            end
        else
            reward_outcomes(t) = observations.rewards(t);
        end

        o = double(actualreward(t) > 0); % win indicator

        % ---- state PE update: pW(selected,start) ----
        if o==1
            pW(selected,1) = pW(selected,1) + params.eta_d_win  * (1 - pW(selected,1));
        else
            pW(selected,1) = pW(selected,1) + params.eta_d_loss * (0 - pW(selected,1));
        end
        pW(selected,1) = clip01(pW(selected,1));

        % ---- forget opposite DIRECT in probability space ----
        opposite = 5 - selected;
        if FORGETopposite
            if FORGETtoZero
                p_target = 0.5; % neutral in probability space
                omega = (pW(opposite,1) >= 0.5)*params.omega_d_posi + (pW(opposite,1) < 0.5)*params.omega_d_nega;
            else
                p_target = pW_init(opposite,1);
                omega = (pW_init(opposite,1) <= pW(opposite,1))*params.omega_d_posi + (pW_init(opposite,1) > pW(opposite,1))*params.omega_d_nega;
            end
            pW(opposite,1) = pW(opposite,1) + omega*(p_target - pW(opposite,1));
            pW(opposite,1) = clip01(pW(opposite,1));
        end

        % ---- As in the MF code: when making a direct choice, also forget advisor-related beliefs (hint-state) via omega_a ----
        for a = 2:3
            if FORGETtoZero
                p_target = 0.5;
                omega = (pW(a,2) >= 0.5)*params.omega_a_posi + (pW(a,2) < 0.5)*params.omega_a_nega;
            else
                p_target = pW_init(a,2);
                omega = (pW_init(a,2) <= pW(a,2))*params.omega_a_posi + (pW_init(a,2) > pW(a,2))*params.omega_a_nega;
            end
            pW(a,2) = pW(a,2) + omega*(p_target - pW(a,2));
            pW(a,2) = clip01(pW(a,2));
        end
        % Enforce symmetry
        pW(2,3) = pW(3,2);
        pW(3,3) = pW(2,2);

        % ---- rebuild Q-values for next trial ----
        qvalue(:,:,t+1) = build_q_from_p(pW, Phint, Vwin_start, Vloss_start, Vwin_hint, Vloss_hint);

    else
        % =========================
        % Ask advice (2-stage)
        % =========================
        if sim == 1
            hint_outcomes(t) = find(rand < cumsum(true_p_o_hint(:,t)'),1);
        else
            hint_outcomes(t) = observations.hints(t);
        end
        hint = hint_outcomes(t)+1; % 2=LH, 3=RH

        action_probs(:,2,t) = [0; spm_softmax(params.inv_temp*qvalue(2:3,hint,t))]';

        if sim == 1
            actions(t,2) = find(rand < cumsum(action_probs(:,2,t)'),1);
        else
            actions(t,2) = choices(t,2);
        end

        if sim == 1
            reward_outcomes(t) = find(rand < cumsum(true_p_o_win(:,actions(t,2),t)'),1);
            if reward_outcomes(t) == 1
                actualreward(t) = 2;
            else
                actualreward(t) = (task.block_type=="SL")*(-4) + (task.block_type=="LL")*(-8);
            end
        else
            reward_outcomes(t) = observations.rewards(t);
        end

        o = double(actualreward(t) > 0);

        secchoice = actions(t,2);
        secopposite = 5 - secchoice;

        % ---- state PE update: pW(secchoice, hintState) ----
        % This is where "trust" (as a win-probability belief in hint states) can be gained/lost over trials.
        if o==1
            pW(secchoice,hint) = pW(secchoice,hint) + params.eta_a_win  * (1 - pW(secchoice,hint));
        else
            pW(secchoice,hint) = pW(secchoice,hint) + params.eta_a_loss * (0 - pW(secchoice,hint));
        end
        pW(secchoice,hint) = clip01(pW(secchoice,hint));

        % If you want generalization from advice trials to start-state beliefs, enable this switch:
        if UPDATE_START_DIRECT_CHOICE
            if o==1
                pW(secchoice,1) = pW(secchoice,1) + params.eta_d_win  * (1 - pW(secchoice,1));
            else
                pW(secchoice,1) = pW(secchoice,1) + params.eta_d_loss * (0 - pW(secchoice,1));
            end
            pW(secchoice,1) = clip01(pW(secchoice,1));
        end

        % ---- forget opposite: start uses omega_d, hint uses omega_a ----
        if FORGETopposite
            % start
            if FORGETtoZero
                p_target = 0.5;
                omega = (pW(secopposite,1) >= 0.5)*params.omega_d_posi + (pW(secopposite,1) < 0.5)*params.omega_d_nega;
            else
                p_target = pW_init(secopposite,1);
                omega = (pW_init(secopposite,1) <= pW(secopposite,1))*params.omega_d_posi + (pW_init(secopposite,1) > pW(secopposite,1))*params.omega_d_nega;
            end
            pW(secopposite,1) = pW(secopposite,1) + omega*(p_target - pW(secopposite,1));
            pW(secopposite,1) = clip01(pW(secopposite,1));

            % hint
            if FORGETtoZero
                p_target = 0.5;
                omega = (pW(secopposite,hint) >= 0.5)*params.omega_a_posi + (pW(secopposite,hint) < 0.5)*params.omega_a_nega;
            else
                p_target = pW_init(secopposite,hint);
                omega = (pW_init(secopposite,hint) <= pW(secopposite,hint))*params.omega_a_posi + (pW_init(secopposite,hint) > pW(secopposite,hint))*params.omega_a_nega;
            end
            pW(secopposite,hint) = pW(secopposite,hint) + omega*(p_target - pW(secopposite,hint));
            pW(secopposite,hint) = clip01(pW(secopposite,hint));
        end

        % symmetry (MF style)
        pW(2, 5-hint) = pW(3, hint);
        pW(3, 5-hint) = pW(2, hint);

        % ---- rebuild qvalue ----
        qvalue(:,:,t+1) = build_q_from_p(pW, Phint, Vwin_start, Vloss_start, Vwin_hint, Vloss_hint);
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

% =========================
% Helpers
% =========================
function qtab = build_q_from_p(pW, Phint, VwS, VlS, VwH, VlH)
% Build Q-values from win-probability beliefs (model-based expected value).
qtab = zeros(3,3);

% start state
qtab(2,1) = pW(2,1)*VwS + (1-pW(2,1))*VlS;
qtab(3,1) = pW(3,1)*VwS + (1-pW(3,1))*VlS;

% hint states (LH=col2, RH=col3)
qtab(2,2) = pW(2,2)*VwH + (1-pW(2,2))*VlH;
qtab(3,2) = pW(3,2)*VwH + (1-pW(3,2))*VlH;

% symmetry convention (same as MF code)
qtab(2,3) = qtab(3,2);
qtab(3,3) = qtab(2,2);

% advisor value via 1-step planning under Phint
v2 = max([qtab(2,2), qtab(3,2)]);
v3 = max([qtab(2,3), qtab(3,3)]);
qtab(1,1) = Phint(1)*v2 + Phint(2)*v3;

end

function x = clip01(x)
% Keep probabilities strictly within (0,1) for numerical stability.
x = max(min(x, 1-1e-6), 1e-6);
end