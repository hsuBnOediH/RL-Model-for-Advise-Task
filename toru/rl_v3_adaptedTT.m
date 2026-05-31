function [results] = rl_v3_adaptedTT(task, MDP, params, sim)
% RL_Advice_Model_TT  Model-based RL advice model adapted for the pipeline.
%
% Adapted from clean_model_based_bandit_sim_v3_trait_params_plotfixed_v3.m
% to match the function interface used by Simple_Formal_Advice_Model_TT.
%
% INPUTS
%   task   - struct with fields:
%               block_type  : "SL" or "LL"
%               true_p_right: [1 x 30] true P(right is correct) per trial
%               true_p_a    : [1 x 30] true advisor accuracy per trial
%   MDP    - struct array (1 x 30), each element has field .o (outcome matrix)
%   params - struct of model parameters (see parameter mapping below)
%   sim    - 0 = fit mode (read choices/outcomes from MDP)
%            1 = simulate mode (sample choices/outcomes from environment)
%
% PARAMETER MAPPING (params struct → internal variables)
%   params.p_a              → initial advisor trust belief
%   params.inv_temp         → inverse temperature (both stages)
%   params.eta_d_win        → direct-choice learning rate after wins
%   params.eta_d_loss       → direct-choice learning rate after losses
%   params.eta_a_win        → trust learning rate after inferred good advice
%   params.eta_a_loss       → trust learning rate after inferred bad advice
%   params.l_loss_value     → loss aversion (scales negative utility)
%   params.Rsensitivity     → extra loss-aversion multiplier in LL blocks
%   params.omega            → (optional) trust forgetting rate on no-hint trials
%   params.self_reliance_bonus → (optional) cost subtracted from value of asking
%
% OUTPUT
%   results.blockwise(1).action_probs  [3 x 2 x nTrials]
%       (:,1,t) = [P(Ask); P(Direct Left); P(Direct Right)] at stage 1
%       (:,2,t) = [0; P(Left); P(Right)] at stage 2 (only on hint trials)

% =========================================================================
% SETUP
% =========================================================================

nTrials = 30;

% --- Reward structure ---
game.winDirect    = 40;
game.winAfterHint = 20;

if task.block_type == "SL" % small loss room
    game.lossAmount = -40;
elseif task.block_type == "LL" % large loss room
    game.lossAmount = -80;
end

valueScale = 40;

% --- Map params to internal variables ---
beliefRight_init = 0.5;          % fixed prior (not fitted)
beliefTrust_init = params.p_a;   % initial trust in advisor

% Different eta resolutions
if isfield(params, 'eta')
    lrDirectWin       = params.eta;
    lrDirectLoss      = params.eta;
    lrTrustGoodAdvice = params.eta;
    lrTrustBadAdvice  = params.eta; 
else
    if isfield(params, 'eta_d')
        lrDirectWin       = params.eta_d;
        lrDirectLoss      = params.eta_d;
    elseif isfield(params, 'eta_d_win')
        lrDirectWin       = params.eta_d_win;
        lrDirectLoss      = params.eta_d_loss;
    end

    if isfield(params, 'eta_a')
        lrTrustGoodAdvice = params.eta_a;
        lrTrustBadAdvice  = params.eta_a;
    elseif isfield(params, 'eta_a_win')
        lrTrustGoodAdvice = params.eta_a_win;  
        lrTrustBadAdvice  = params.eta_a_loss;  
    end
end

invTemp1 = params.inv_temp;   % stage-1 inverse temperature
invTemp2 = params.inv_temp;   % stage-2 inverse temperature (same param)

% Loss aversion:
if task.block_type == "LL"
    lossAversion = params.Rsensitivity;
else
    lossAversion = params.l_loss_value;
end

% Optional trust forgetting on no-hint trials
if isfield(params, 'omega') && params.omega > 0
    applyTrustForgetting = true;
    trustForgetRate = params.omega;
else
    applyTrustForgetting = false;
    trustForgetRate = 0;
end

% Optional advice cost (self-reliance bonus)
if isfield(params, 'self_reliance_bonus')
    adviceCost = params.self_reliance_bonus;
else
    adviceCost = 0;
end

% Build a local agent struct for computeActionValues
agent.lossAversion = lossAversion;
agent.adviceCost   = adviceCost;

% =========================================================================
% PARSE INPUTS (sim = 0: read observed choices and outcomes from MDP)
% =========================================================================
% Encoding matches Simple_Formal_Advice_Model_TT:
%   observations.hints(t) : 0 = no hint, 1 = left hint, 2 = right hint
%   observations.rewards(t): 1 = win,    2 = loss
%   choices(t,1): 1 = asked advisor, 2 = chose left, 3 = chose right
%   choices(t,2): 2 = left (after hint), 3 = right (after hint), 0 = n/a

observations.hints   = zeros(1, nTrials);
observations.rewards = zeros(1, nTrials);
choices              = zeros(nTrials, 2);

if sim == 0
    for trial = 1:nTrials
        trial_info = MDP(trial);
        observations.hints(trial) = trial_info.o(1,2) - 1;

        if observations.hints(trial)   % hint trial (advisor was asked)
            observations.rewards(trial) = 4 - trial_info.o(2,3);  % 1=win, 2=loss
            choices(trial, 1) = 1;                                 % asked advisor
            choices(trial, 2) = trial_info.o(3,3) - 1;            % 2=left, 3=right
        else                           % direct trial
            observations.rewards(trial) = 4 - trial_info.o(2,2);  % 1=win, 2=loss
            choices(trial, 1) = trial_info.o(3,2) - 1;            % 2=left, 3=right
            choices(trial, 2) = 0;                                 % n/a
        end
    end
end

% =========================================================================
% PREALLOCATE
% =========================================================================

beliefRight = nan(1, nTrials + 1);
beliefTrust = nan(1, nTrials + 1);
beliefRight(1) = clip01(beliefRight_init);
beliefTrust(1) = clip01(beliefTrust_init);

policyStart      = nan(nTrials, 3);
policyAfterLeft  = nan(nTrials, 2);
policyAfterRight = nan(nTrials, 2);

% action_probs: [3 x 2 x nTrials]
%   dim 1: action index (1=Ask/Left/Right, 2=Left, 3=Right)
%   dim 2: stage (1 = stage 1, 2 = stage 2)
%   dim 3: trial
action_probs = zeros(3, 2, nTrials);

stage1Action      = nan(1, nTrials);
secondStageAction = nan(1, nTrials);
askedForHint      = false(1, nTrials);
hintWasRight      = nan(1, nTrials);
wonTrial          = false(1, nTrials);
rewardReceived    = nan(1, nTrials);
finalChoice       = nan(1, nTrials);   % 0 = Left, 1 = Right

% =========================================================================
% MAIN TRIAL LOOP
% =========================================================================

for t = 1:nTrials

    currentBeliefRight = beliefRight(t);
    currentBeliefTrust = beliefTrust(t);

    % --- Compute model-based action values from current beliefs ---
    [valueAsk, valueLeftDirect, valueRightDirect, ...
     vIfLeftHint_chooseLeft,  vIfLeftHint_chooseRight, ...
     vIfRightHint_chooseLeft, vIfRightHint_chooseRight] = ...
        computeActionValues(currentBeliefRight, currentBeliefTrust, game, agent);

    % --- Softmax policies ---
    policyStart(t,:) = spm_softmax((invTemp1 / valueScale) * ...
    [valueAsk; valueLeftDirect; valueRightDirect])';

    policyAfterLeft(t,:) = spm_softmax((invTemp2 / valueScale) * ...
    [vIfLeftHint_chooseLeft; vIfLeftHint_chooseRight])';

    policyAfterRight(t,:) = spm_softmax((invTemp2 / valueScale) * ...
    [vIfRightHint_chooseLeft; vIfRightHint_chooseRight])';

    % Store stage-1 probabilities
    action_probs(1, 1, t) = policyStart(t, 1);   % P(Ask)
    action_probs(2, 1, t) = policyStart(t, 2);   % P(Direct Left)
    action_probs(3, 1, t) = policyStart(t, 3);   % P(Direct Right)

    % --- Stage-1 action ---
    if sim == 1
        trueRightCorrect  = rand < task.true_p_right(t);
        stage1Action(t)   = sampleCategorical(policyStart(t,:));
    else
        stage1Action(t)   = choices(t, 1);   % 1=Ask, 2=Left, 3=Right
    end

    % -------------------------------------------------------------------------
    if stage1Action(t) == 1
        % ASKED FOR HINT
        askedForHint(t) = true;

        % Determine which hint was given (simulated or observed)
   if sim == 1
      if rand < task.true_p_a(t)
         hintWasRight(t) = trueRightCorrect;
      else
        hintWasRight(t) = ~trueRightCorrect;
      end

        observations.hints(t) = 1 + double(hintWasRight(t));  % 1=left hint, 2=right hint
   else
       % observations.hints: 1 = left hint, 2 = right hint
       hintWasRight(t) = (observations.hints(t) == 2);
   end

        % Stage-2 policy and action_probs depend on which hint was given
        if hintWasRight(t)
            policyObs2 = policyAfterRight(t,:);
            action_probs(2, 2, t) = policyAfterRight(t, 1);  % P(Left  | right hint)
            action_probs(3, 2, t) = policyAfterRight(t, 2);  % P(Right | right hint)
        else
            policyObs2 = policyAfterLeft(t,:);
            action_probs(2, 2, t) = policyAfterLeft(t, 1);   % P(Left  | left hint)
            action_probs(3, 2, t) = policyAfterLeft(t, 2);   % P(Right | left hint)
        end

        % Stage-2 action
        if sim == 1
            secondStageAction(t) = sampleCategorical(policyObs2);  % 1=Left, 2=Right
        else
            % choices(t,2): 2=left, 3=right → subtract 1 → 1=Left, 2=Right
            secondStageAction(t) = choices(t, 2) - 1;
        end

        finalChoice(t) = secondStageAction(t) - 1;   % 0=Left, 1=Right

        % Outcome
        if sim == 1
            wonTrial(t) = (finalChoice(t) == trueRightCorrect);
        else
            wonTrial(t) = (observations.rewards(t) == 1);
        end

        if wonTrial(t)
            rewardReceived(t) = game.winAfterHint;
        else
            rewardReceived(t) = game.lossAmount;
        end

        % Did participant follow the hint?
        followedHint = (finalChoice(t) == hintWasRight(t));

        % --- Belief updates ---

        % (i) Direct-choice correctness belief — updated every trial
        rightWasCorrectInferred = inferRightWasCorrect(finalChoice(t), wonTrial(t));
        if wonTrial(t), lrDirect = lrDirectWin; else, lrDirect = lrDirectLoss; end
        beliefRight(t+1) = deltaUpdate(currentBeliefRight, rightWasCorrectInferred, lrDirect);

        % (ii) Advisor trust — updated only on hint trials
        hintWasCorrectInferred = (followedHint && wonTrial(t)) || ...
                                 (~followedHint && ~wonTrial(t));
        if hintWasCorrectInferred
            lrTrust = lrTrustGoodAdvice;
        else
            lrTrust = lrTrustBadAdvice;
        end
        beliefTrust(t+1) = deltaUpdate(currentBeliefTrust, double(hintWasCorrectInferred), lrTrust);

    % -------------------------------------------------------------------------
    else
        % DIRECT CHOICE (no advisor)
        askedForHint(t) = false;

        if stage1Action(t) == 2
            finalChoice(t) = 0;   % Left
        else
            finalChoice(t) = 1;   % Right
        end

        % Outcome
        if sim == 1
            wonTrial(t) = (finalChoice(t) == trueRightCorrect);
        else
            wonTrial(t) = (observations.rewards(t) == 1);
        end

        if wonTrial(t)
            rewardReceived(t) = game.winDirect;
        else
            rewardReceived(t) = game.lossAmount;
        end

        % --- Belief updates ---

        % (i) Direct-choice correctness belief updated every trial
        rightWasCorrectInferred = inferRightWasCorrect(finalChoice(t), wonTrial(t));
        if wonTrial(t), lrDirect = lrDirectWin; else, lrDirect = lrDirectLoss; end
        beliefRight(t+1) = deltaUpdate(currentBeliefRight, rightWasCorrectInferred, lrDirect);

        % (ii) Trust unchanged on direct trials (optional forgetting below)
        beliefTrust(t+1) = currentBeliefTrust;
    end

    % --- Optional trust forgetting toward initial prior on no-hint trials ---
    if applyTrustForgetting && ~askedForHint(t)
        beliefTrust(t+1) = beliefTrust(t+1) + ...
            trustForgetRate * (beliefTrust_init - beliefTrust(t+1));
    end

    beliefRight(t+1) = clip01(beliefRight(t+1));
    beliefTrust(t+1) = clip01(beliefTrust(t+1));

end  % end trial loop

% =========================================================================
% BUILD OUTPUT STRUCTURE
% Matches the format expected by advice_inversionTT and Advice_fit_prolificTT
% =========================================================================

results.blockwise(1).action_probs    = action_probs;
actions_out = [stage1Action; secondStageAction + 1]';
actions_out(~askedForHint, 2) = 0;
results.blockwise(1).actions = actions_out;
results.blockwise(1).hint_outcomes = observations.hints;
results.blockwise(1).reward_outcomes = 2 - double(wonTrial);   % 1=win, 2=loss
results.blockwise(1).actualreward    = rewardReceived;
results.blockwise(1).belief_right    = beliefRight;
results.blockwise(1).belief_trust    = beliefTrust;
results.blockwise(1).policy_start    = policyStart;
results.blockwise(1).policy_after_left  = policyAfterLeft;
results.blockwise(1).policy_after_right = policyAfterRight;

end  % end main function


% =========================================================================
% LOCAL HELPER FUNCTIONS
% (Carried over from clean_model_based_bandit_sim_v3_trait_params_plotfixed_v3)
% =========================================================================

function x = clip01(x)
% Keep a probability strictly inside (eps, 1-eps)
    x = min(max(x, 1e-6), 1 - 1e-6);
end


function idx = sampleCategorical(p)
% Sample index 1..K from probability vector p
    idx = find(rand < cumsum(p), 1, 'first');
end

function updatedBelief = deltaUpdate(oldBelief, target, lr)
% Standard delta-rule update for a probability belief
    updatedBelief = oldBelief + lr * (target - oldBelief);
    updatedBelief = clip01(updatedBelief);
end

function rightWasCorrect = inferRightWasCorrect(chosenRight, wonTrial)
% Infer whether Right was the correct option from choice and outcome.
%   chosenRight : 1 if Right was chosen, 0 if Left
%   wonTrial    : 1/true if trial was won
    if (chosenRight == 1 && wonTrial == 1) || (chosenRight == 0 && wonTrial == 0)
        rightWasCorrect = 1;
    else
        rightWasCorrect = 0;
    end
end

function u = utilityOfReward(r, lossAversion)
% Subjective utility
    if r >= 0
        u = r;
    else
        u = -lossAversion * 40;
    end
end

function [valueAsk, valueLeftDirect, valueRightDirect, ...
          vIfLeftHint_chooseLeft,  vIfLeftHint_chooseRight, ...
          vIfRightHint_chooseLeft, vIfRightHint_chooseRight] = ...
          computeActionValues(beliefRightCorrect, beliefHintTrust, game, agent)
% Compute model-based expected values from current beliefs.

    pRight    = beliefRightCorrect;
    pLeft     = 1 - pRight;
    pTrust    = beliefHintTrust;
    pDistrust = 1 - pTrust;

    UwinDirect    = utilityOfReward(game.winDirect,    agent.lossAversion);
    UwinAfterHint = utilityOfReward(game.winAfterHint, agent.lossAversion);
    Uloss         = utilityOfReward(game.lossAmount,   agent.lossAversion);

    % Direct choices from start state
    valueLeftDirect  = pLeft  * UwinDirect + (1 - pLeft)  * Uloss;
    valueRightDirect = pRight * UwinDirect + (1 - pRight)  * Uloss;

    % After a LEFT hint: choose Left = follow, choose Right = ignore
    vIfLeftHint_chooseLeft  = pTrust    * UwinAfterHint + (1 - pTrust)    * Uloss;
    vIfLeftHint_chooseRight = pDistrust * UwinAfterHint + (1 - pDistrust) * Uloss;

    % After a RIGHT hint: symmetric
    vIfRightHint_chooseRight = vIfLeftHint_chooseLeft;
    vIfRightHint_chooseLeft  = vIfLeftHint_chooseRight;

    % Asking for advice: assume P(left hint) = P(right hint) = 0.5
    bestAfterLeftHint  = max(vIfLeftHint_chooseLeft,   vIfLeftHint_chooseRight);
    bestAfterRightHint = max(vIfRightHint_chooseLeft,  vIfRightHint_chooseRight);
    valueAsk = 0.5 * bestAfterLeftHint + 0.5 * bestAfterRightHint - agent.adviceCost;
end