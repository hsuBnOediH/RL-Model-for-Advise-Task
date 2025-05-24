
function [DCM] = advice_inversion_uni(DCM, model)
    ALL = false;
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
                    'omega_d_win', 'eta_d_loss', 'omega_d_loss', 'lamgda'})
                pE.(field) = log(DCM.params.(field)/(1-DCM.params.(field)));  % bound between 0 and 1
                pC{i,i}    = prior_variance;
            elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
                    'parameter_exploration', 'Rsensitivity'})
                pE.(field) = log(DCM.params.(field));               % in log-space (to keep positive)
                pC{i,i}    = prior_variance;  
            else
                pE.(field) = DCM.params.(field); 
                pC{i,i}    = prior_variance;
            end
        end
    end

    pC = spm_cat(pC);

    % model specification
    %--------------------------------------------------------------------------
    M.model = model;
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
                'omega_d_win', 'eta_d_loss', 'omega_d_loss', 'lamgda'})
            params.(field) = 1/(1+exp(-P.(field)));
        elseif ismember(field, {'inv_temp', 'reward_value', 'l_loss_value', 'state_exploration',...
                'parameter_exploration', 'Rsensitivity'})
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
        % TODO: if the advise gen_mode is still needed
        % TODO: what is the MDP before here? exist or not? 
        % TODO: is the params need to be change?
        %  priors = 
        %    struct with fields:
        %    p_ha: 0.7500
        %    omega_eta_advisor_win: 0.6000
        %    omega_eta_advisor_loss: 0.6000
        %    omega_eta_context: 0.6000
        %    novelty_scalar: 0.3000
        %    alpha: 2
        % trialinfo(30*idx_block-29:30*idx_block,:)
        %    30x3 cell array
        %    {'0.9'}    {'0.4'}    {'40'}
        %    {'0.9'}    {'0.4'}    {'40'}
        if M.model == 4
            MDP     = advise_gen_model_uni(trialinfo(30*idx_block-29:30*idx_block,:),params);
        end
        % after advise_gen_model, MDP is a 1x30 struct array with fields:
        % MDP =
        %            T: 3
        %            V: [2x4x2 double]
        %            A: {[3x2x4 double]  [3x2x4 double]  [4x2x4 double]}
        %            B: {[2x2 double]  [4x4x4 double]}
        %            C: {[3x3 double]  [3x3 double]  [4x3 double]}
        %            D: {[2x1 double]  [4x1 double]}
        %            d: {[2x1 double]  [4x1 double]}
        %    omega_eta_advisor_win: 0.6000
        %    omega_eta_advisor_loss: 0.6000
        %    omega_eta_context: 0.6000
        %        alpha: 2
        %        beta: 1
        %        erp: 1
        %        tau: 2
        %    prior_d: 1
        %        p_ha: 0.7500
        %    prior_a: 1
        %        rs: 1
        %    novelty_scalar: 0.3000
        %            a: {[3x2x4 double]  [3x2x4 double]  [4x2x4 double]}
        %    a_floor: [3x2 double]
        %    d_floor: [2x1 double]
        %        label: [1x1 struct]
        % TODO: where to define the policy?


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
            % outcomes{1}
                % 1     3     1
                % 1     1     3
                % 1     2     4
            actions  = Y(30*idx_block-29:30*idx_block);
            % actions{1}
            %     1     1
            %     2     4
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






        if M.model == 1
            MDP  = active_inference_model_uni(task, MDP,params, 0);
        elseif M.model == 2
            MDP  = rl_model_connect_uni(task, MDP,params, 0);
        elseif M.model == 3
            MDP  = rl_model_disconnect_uni(task, MDP,params, 0);
        elseif M.model == 4
            MDP  = active_inference_model_mp_uni(task, MDP,params, 0);
            % after MDP is   1x30 struct array with fields:
            %                T: 3
            %                V: [2x4x2 double]
            %                A: {[3x2x4 double]  [3x2x4 double]  [4x2x4 double]}
            %                B: {[2x2 double]  [4x4x4 double]}
            %                C: {[3x3 double]  [3x3 double]  [4x3 double]}
            %                D: {[2x1 double]  [4x1 double]}
            %                d: {[2x1 double]  [4x1 double]}
            %    omega_eta_advisor_win: 0.6000
            %     omega_eta_advisor_loss: 0.6000
            %    omega_eta_context: 0.6000
            %            alpha: 2
            %            beta: 1
            %            erp: 1
            %            tau: 2
            %        prior_d: 1
            %            p_ha: 0.7500
            %        prior_a: 1
            %            rs: 1
            %    novelty_scalar: 0.3000
            %                a: {[3x2x4 double]  [3x2x4 double]  [4x2x4 double]}
            %        a_floor: [3x2 double]
            %        d_floor: [2x1 double]
            %            label: [1x1 struct]
            %                o: [3x3 double]
            %                u: [2x2 double]
            %                s: [2x3 double]
            %                F: [4x3 double]
            %                G: [4x3 double]
            %                H: [1 1 1]
            %            Fa: [-0.0358 0 0]
            %            Fd: [-0.1090 0]
            %                O: {3x3 cell}
            %                P: [1x4x2 double]
            %                R: [4x3 double]
            %                Q: {[2x3x4 double]  [4x3x4 double]}
            %                X: {[2x3 double]  [4x3 double]}
            %                w: [1 1 1]
            %            vn: {[1x2x3x3 double]  [1x4x3x3 double]}
            %            xn: {[1x2x3x3 double]  [1x4x3x3 double]}
            %            un: [4x3 double]
            %            wn: [3x1 double]
            %            dn: [3x1 double]
            %            rt: [0.0439 0.0237 0.0066]

        else
            print('Model not recognized')
        end


        for j = 1:block_size
            if actions{j}(2,1) ~= 2
                % TODO: still use the old model? to compute the log likelihood?
                if M.model == 4
                   L = L + log(MDP(j).P(1,actions{j}(2,1),1) + eps); %old model
                else
                    %L = L + log(MDP(j).P(1,actions{j}(2,1),1) + eps); old model
                    prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,1)-1,1,j); 
                    L = L + log(prob_choose_bandit + eps);
                end
            else % when advisor was chosen
                if M.model == 4
                    prob_choose_advisor = MDP(j).P(1,actions{j}(2,1),1);
                    L = L + log(prob_choose_advisor + eps);
                    prob_choose_bandit = MDP(j).P(1,actions{j}(2,2),2);
                    L = L + log(prob_choose_bandit + eps);
                else
                    prob_choose_advisor = MDP.blockwise.action_probs(1,1,j); 
                    %prob_choose_advisor = MDP(j).P(1,actions{j}(2,1),1); old model
                    L = L + log(prob_choose_advisor + eps);
                    prob_choose_bandit = MDP.blockwise.action_probs(actions{j}(2,2)-1,2,j); 
                    %prob_choose_bandit = MDP(j).P(1,actions{j}(2,2),2); old model
                    L = L + log(prob_choose_bandit + eps);
                end
            end
        end

        clear('MDP')


    end

    fprintf('LL: %f \n',L)
return


