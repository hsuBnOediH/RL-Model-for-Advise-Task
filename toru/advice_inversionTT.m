% Samuel Taylor and Ryan Smith, 2021
% Model inversion script
function [DCM] = advice_inversionTT(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
prior_variance = .5;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        % transform the parameters that we fit
        if ismember(field, {'p_right', 'p_a', 'eta', 'omega', 'eta_a_win', 'omega_a_win',...
                'eta_a','omega_a','eta_d','omega_d','eta_a_loss','omega_a_loss','eta_d_win'...
                'omega_d_win', 'eta_d_loss', 'omega_d_loss', 'eta'})
            pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));  % bound between 0 and 1
            pC{i,i}    = prior_variance;
        elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
                'parameter_exploration', })
            pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
            pC{i,i}    = prior_variance;  
        else
            pE.(field) = DCM.params.(field); 
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
%M.mdp   = DCM.MDP;                       % MDP structure
M.mode  = DCM.mode;
M.trialinfo = DCM.trialinfo;
M.params = DCM.params;
M.actualrewards =DCM.actualrewards;

% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidence (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return

function L = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% multiply parameters in MDP
%--------------------------------------------------------------------------
%mdp   = M.mdp;
fields = fieldnames(M.pE);
params = M.params;
for i = 1:length(fields)
    field = fields{i};
    if ismember(field, {'p_right', 'p_a', 'eta', 'omega', 'eta_a_win', 'omega_a_win',...
            'eta_a','omega_a','eta_d','omega_d','eta_a_loss','omega_a_loss','eta_d_win'...
            'omega_d_win', 'eta_d_loss', 'omega_d_loss'})
        params.(field) = 1/(1+exp(-P.(field)));
    elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
            'parameter_exploration', })
        params.(field) = exp(P.(field));
    else
        params.(field) = P.(field);
    end
end



% discern whether learning is enabled - and identify unique trials if not
%--------------------------------------------------------------------------
if any(ismember(fieldnames(params),{'a','b','d','c','d','e'}))
    j = 1:numel(U);
    k = 1:numel(U);
else
    % find unique trials (up until the last outcome)
    %----------------------------------------------------------------------
    u       = spm_cat(U');
    [i,j,k] = unique(u(:,1:(end - 1)),'rows');
end

num_trials = size(U,2);
num_blocks = floor(num_trials/30);
if num_trials == 1
    block_size = 1;
else
    block_size = 30;
end

trialinfo = M.trialinfo;
L = 0;

% Each block is separate -- effectively resetting beliefs at the start of
% each block. 
for idx_block = 1:num_blocks
    % note that this generative model is outdated for new advise task model
    %MDP     = advise_gen_model(trialinfo(30*idx_block-29:30*idx_block,:),params);
    %[MDP(1:block_size)]   = deal(mdp_block);
    if (num_trials == 1)
        outcomes = U;
        actions = Y;
        MDP.o  = outcomes{1};
        MDP.u  = actions{1};
        MDP.actualreward  = M.actualrewards(1);
    else
        outcomes = U(30*idx_block-29:30*idx_block);
        actions  = Y(30*idx_block-29:30*idx_block);
        actualreward = M.actualrewards(30*idx_block-29:30*idx_block);
        task.true_p_right = nan(1,30);
        for idx_trial = 1:30
            MDP(idx_trial).o = outcomes{idx_trial};
            MDP(idx_trial).u = actions{idx_trial};
            MDP(idx_trial).actualreward = actualreward(idx_trial);
            task.true_p_right(idx_trial) = 1-str2double(trialinfo{(idx_block-1)*30+idx_trial,2});
            task.true_p_a(idx_trial) = str2double(trialinfo{(idx_block-1)*30+idx_trial,1});

        end
        if strcmp(trialinfo{idx_block*30-29,3}, '80')
            task.block_type = "LL";
        else
            task.block_type = "SL";
        end
        
    end
    
    % solve MDP and accumulate log-likelihood
    %--------------------------------------------------------------------------
    
    %MDP  = spm_MDP_VB_X_advice(MDP);
    %MDP  = spm_MDP_VB_X_advice_no_message_passing_faster(MDP);
    task.field = fields;
%    MDP  = Simple_Advice_Model_TT(task, MDP,params, 0);
    MDP  = ModelFreeRLModel_TT(task, MDP,params, 0);

    for j = 1:block_size
        if actions{j}(2,1) ~= 2
            %L = L + log(MDP(j).P(1,actions{j}(2,1),1) + eps); old model
            prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,1)-1,1,j); 
            L = L + log(prob_choose_bandit + eps);
            
        else % when advisor was chosen
            prob_choose_advisor = MDP.blockwise.action_probs(1,1,j); 
            %prob_choose_advisor = MDP(j).P(1,actions{j}(2,1),1); old model
            L = L + log(prob_choose_advisor + eps);
            prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,2)-1,2,j); 
            %prob_choose_bandit = MDP(j).P(1,actions{j}(2,2),2); old model
            L = L + log(prob_choose_bandit + eps);
        end
    end

    clear('MDP')


end

fprintf('LL: %f \n',L)


