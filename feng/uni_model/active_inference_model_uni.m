function [results] = active_inference_model_uni(task, MDP, params, sim)

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
    a_floor = 1;
    context_floor = 1;
    
    block = 1;
    clear Q
    clear epistemic_value
    clear pragmatic_value
    clear novelty_value
    
    hint_outcomes(1,1:task.num_trials) = 0;

    for trial = 1:task.num_trials
        hint_outcome_vector(:,trial) = [0 0]';
        dir_context(:,:,trial) = zeros(2,1);
        p_context(:,:,trial) = zeros(2,1);
        pp_context(:,:,trial) = zeros(2,1);
        true_context(:,:,trial) = zeros(2,1);
        a{1}(:,:,trial) = zeros(2,2);
        true_A{1}(:,:,trial) = zeros(2,2);
        A{2}(:,:,trial) = zeros(2,2);
        action_probs(:,:,trial) = zeros(3,2);
        Q(:,:,trial) = zeros(3,1);
        epistemic_value(:,:,trial) = zeros(3,1);
        pragmatic_value(:,:,trial) = zeros(3,1);
        novelty_value(:,:,trial) = zeros(3,1);
        p_o_win(:,:,trial) = zeros(2,3);
        if sim == 1
            true_context_vector(trial) = find(rand < cumsum([1-task.true_p_right(blocsk,trial) task.true_p_right(block,trial)]'),1)-1;
        end
    end
 

    d_0 = [1-params.p_right params.p_right]'*context_floor;
    a_0 =  [params.p_a 1-params.p_a;         % "try left"
            1-params.p_a params.p_a]*a_floor;% "try right"   

    actions = zeros(task.num_trials,2);
    
    % reward value distribution
    if task.block_type(block)== "LL"
        R(:,block) =  spm_softmax([params.reward_value+eps (-params.l_loss_value*params.Rsensitivity)-eps]');
        Rafteradvice(:,block) = spm_softmax([params.reward_value+eps (-params.l_loss_value*params.Rsensitivity)-eps]');
    else
        R(:,block) =  spm_softmax([params.reward_value+eps -params.l_loss_value-eps]');
        Rafteradvice(:,block) =  spm_softmax([params.reward_value+eps -params.l_loss_value-eps]');
    end

    if sim == 0
        for trial = 1:task.num_trials
            true_context_vector(trial) = task.true_p_right(block,trial);
        end
    end

    for trial = 1:task.num_trials
        tp = 1; % initial timepoint
        % priors
        if trial == 1
            dir_context(:,:,trial) = d_0;
        end 
        p_context(:,:,trial) = spm_norm(dir_context(:,:,trial));
        true_context(:,:,trial) = [1-true_context_vector(trial) true_context_vector(trial)]';
            
        % likelihood mapping
        %hint (concentration parameters)
        if trial == 1
            a{1}(:,:,trial) = a_0;
        end
        % p(hint|context)
        A{1}(:,:,trial) = spm_norm(a{1}(:,:,trial));
        true_A{1}(:,:,trial) =  [task.true_p_a(block,trial)   1-task.true_p_a(block,trial); % "try left"
                                 1-task.true_p_a(block,trial) task.true_p_a(block,trial)];  % "try right"
        %left better
        A{2}(:,:,1) = [p_win   1-p_win; % win
                       1-p_win p_win];  % lose
        %right better
        A{2}(:,:,2) = [1-p_win p_win;   % win
                       p_win   1-p_win];% lose

        for option = 1:3
            if option == 1
                p_o_hint(:,trial) = A{1}(:,:,1)*p_context(:,:,trial);
                true_p_o_hint(:,trial) = true_A{1}(:,:,1)*true_context(:,:,trial);
                % novelty_value(option,tp,trial) = .5*dot(A{1}(:,1,trial),info_gain(:,1)) + .5*dot(A{1}(:,2,trial),info_gain(:,1));
                % novelty_value(option,tp,trial) = (sum(sum(a{1}(:,:,trial))))^-1;
                a_sums{1}(:,:,trial) = [sum(a{1}(:,1,trial)) sum(a{1}(:,2,trial)); sum(a{1}(:,1,trial)) sum(a{1}(:,2,trial))];
                info_gain = (a{1}(:,:,trial).^-1) - (a_sums{1}(:,:,trial).^-1);
                %marginalize over context state factor (i.e. left better or
                %right better)
                novelty_for_each_observation = info_gain(:,1)*p_context(1,:,trial) + info_gain(:,2)*p_context(2,:,trial);
                novelty_value(option,tp,trial) = sum(novelty_for_each_observation);
                epistemic_value(option,tp,trial) = G_epistemic_value(log(A{1}(:,:,trial)),log(p_context(:,:,trial)));
                % epistemic_value(option,tp,trial) = G_epistemic_value(A{1}(:,:,trial),p_context(:,:,trial));
                pragmatic_value(option,tp,trial) = 0;
            elseif option == 2 
                p_o_win(:,option,trial) = A{2}(:,:,1)*p_context(:,:,trial);
                true_p_o_win(:,option,trial) = A{2}(:,:,1)*true_context(:,:,trial);
                novelty_value(option,tp,trial) = 0;
                epistemic_value(option,tp,trial) = 0;
                pragmatic_value(option,tp,trial) = log(dot(p_o_win(:,option,trial),R(:,block)));
                % pragmatic_value(option,tp,trial) = dot(p_o_win(:,option,trial),R(:,block));
            elseif option == 3 
                p_o_win(:,option,trial) = A{2}(:,:,2)*p_context(:,:,trial);
                true_p_o_win(:,option,trial) = A{2}(:,:,2)*true_context(:,:,trial);
                novelty_value(option,tp,trial) = 0;
                epistemic_value(option,tp,trial) = 0;
                pragmatic_value(option,tp,trial) = log(dot(p_o_win(:,option,trial),R(:,block)));
                % pragmatic_value(option,tp,trial) = dot(p_o_win(:,option,trial),R(:,block));
            end
            Q(option, tp,trial) = - params.state_exploration*epistemic_value(option,tp,trial) - pragmatic_value(option,tp,trial) + params.parameter_exploration*novelty_value(option,tp,trial);
            % Q(option, tp,trial) = params.state_exploration*epistemic_value(option,tp,trial) + pragmatic_value(option,tp,trial) + params.parameter_exploration*novelty_value(option,tp,trial);
        end
        % compute action probabilities
        action_probs(:,tp,trial) = spm_softmax(params.inv_temp*(-Q(:,tp,trial)))';
        % action_probs(:,tp,trial) = spm_softmax(params.inv_temp*Q(:,tp,trial))';
        % select actions
        % note that 1 corresponds to choosing advisor, 2 corresponds to
        % choosing left bandit, 3 corresponds to choosing right bandit.
        if sim == 1
            actions(trial,tp) = find(rand < cumsum(action_probs(:,tp,trial)'),1);
        else
            actions(trial,tp) = choices(trial,tp,block);
        end
        % if first action was choosing a bandit, only update context state
        % vector
        if actions(trial,1) ~= 1
            if sim == 1
                hint_outcomes(trial) = 0;
            end
            % hint_outcome_vector(:,trial) = [0 0]';
            % get reward outcome
            if sim == 1
                reward_outcomes(trial) = find(rand < cumsum(true_p_o_win(:,actions(trial,tp),trial)'),1);
            else
                reward_outcomes(trial) = observations.rewards(block,trial);
            end

            if actions(trial,tp) == 3 && reward_outcomes(trial) == 1
                ppp_context(:,trial) = [0 1]';
            elseif actions(trial,tp) == 2 && reward_outcomes(trial) == 2
                ppp_context(:,trial) = [0 1]';
            elseif actions(trial,tp) == 2 && reward_outcomes(trial) == 1
                ppp_context(:,trial) = [1 0]';
            elseif actions(trial,tp) == 3 && reward_outcomes(trial) == 2
                ppp_context(:,trial) = [1 0]';
            end
            
           % make actualrewards for simulation
            if reward_outcomes(trial) == 1
                actualreward(trial) = 40;
            elseif reward_outcomes(trial) == 2
                if task.block_type == "SL"
                    actualreward(trial) = -40;
                elseif task.block_type == "LL"
                    actualreward(trial) = -80;
                end
            end

        % if first action was choosing advisor, update likelihood matrices
        % before picking pandit
        elseif actions(trial,1) == 1
            tp = 2; % second time point
            % get hint outcome
            if sim == 1
                hint_outcomes(trial) = find(rand < cumsum(true_p_o_hint(:,trial)'),1);
            else 
                hint_outcomes(trial) = observations.hints(block,trial);
            end
            hint_outcome_vector(hint_outcomes(trial),trial) = 1;
            % state belief update
            pp_context(:,:,trial) = p_context(:,:,trial).*A{1}(:,hint_outcomes(trial),trial)...
                            /sum(p_context(:,:,trial).*A{1}(:,hint_outcomes(trial),trial),1);
            for option = 2:3
                Q(1, tp,trial) = eps;
                if option == 2 
                    p_o_win(:,option,trial) = A{2}(:,:,1)*pp_context(:,:,trial);
                    novelty_value(option,tp,trial) = 0;
                    epistemic_value(option,tp,trial) = 0;
                    pragmatic_value(option,tp,trial) = dot(p_o_win(:,option,trial),Rafteradvice(:,block));
                elseif option == 3 
                    p_o_win(:,option,trial) = A{2}(:,:,2)*pp_context(:,:,trial);
                    novelty_value(option,tp,trial) = 0;
                    epistemic_value(option,tp,trial) = 0;
                    pragmatic_value(option,tp,trial) = dot(p_o_win(:,option,trial),Rafteradvice(:,block));
                end
                Q(option, tp,trial) = params.state_exploration*epistemic_value(option,tp,trial) + pragmatic_value(option,tp,trial) + params.parameter_exploration*novelty_value(option,tp,trial);
            end
            % compute action probabilities
            action_probs(:,tp,trial) = [0; spm_softmax(params.inv_temp*Q(2:3,tp,trial))]';
            % select actions
            if sim == 1
                actions(trial,tp) = find(rand < cumsum(action_probs(:,tp,trial)'),1);
            else
                actions(trial,tp) = choices(trial,tp,block);
            end
            % get reward outcome
            if sim == 1
                reward_outcomes(trial) = find(rand < cumsum(true_p_o_win(:,actions(trial,tp),trial)'),1);
            else
                reward_outcomes(trial) = observations.rewards(block,trial);
            end    
    
            if actions(trial,tp) == 3 && reward_outcomes(trial) == 1
                ppp_context(:,trial) = [0 1]';
            elseif actions(trial,tp) == 2 && reward_outcomes(trial) == 2
                ppp_context(:,trial) = [0 1]';
            elseif actions(trial,tp) == 2 && reward_outcomes(trial) == 1
                ppp_context(:,trial) = [1 0]';
            elseif actions(trial,tp) == 3 && reward_outcomes(trial) == 2
                ppp_context(:,trial) = [1 0]';
            end  
             
            % make actualrewards for simulation
            if reward_outcomes(trial) == 1
                actualreward(trial) = 20;
            elseif reward_outcomes(trial) == 2
                if task.block_type == "SL"
                    actualreward(trial) = -40;
                elseif task.block_type == "LL"
                    actualreward(trial) = -80;
                end
            end
        end   
             
            
        if reward_outcomes(trial) == 1
            if actions(trial,1) == 1 
                % forgetting part
                a{1}(:,:,trial+1) = (a{1}(:,:,trial) - a_0)*(1-params.omega_a_win) + a_0;
                % learning part
                a{1}(:,:,trial+1) = a{1}(:,:,trial+1) + params.eta_a_win*(ppp_context(:,trial)*hint_outcome_vector(:,trial)')';
            else
                a{1}(:,:,trial+1) = a{1}(:,:,trial);
            end
                % forgetting part
                dir_context(:,:,trial+1) = (dir_context(:,:,trial) - d_0)*(1-params.omega_d_win) + d_0;
                % learning part
                dir_context(:,:,trial+1) = dir_context(:,:,trial+1) + params.eta_d_win*ppp_context(:,trial);
        elseif reward_outcomes(trial) == 2
            if actions(trial,1) == 1 
                % forgetting part
                a{1}(:,:,trial+1) = (a{1}(:,:,trial) - a_0)*(1-params.omega_a_loss) + a_0;
                % learning part
                a{1}(:,:,trial+1) = a{1}(:,:,trial+1) + params.eta_a_loss*(ppp_context(:,trial)*hint_outcome_vector(:,trial)')';
            else
                a{1}(:,:,trial+1) = a{1}(:,:,trial);
            end
            % forgetting part
            dir_context(:,:,trial+1) = (dir_context(:,:,trial) - d_0)*(1-params.omega_d_loss) + d_0;
            % learning part
            dir_context(:,:,trial+1) = dir_context(:,:,trial+1) + params.eta_d_loss*ppp_context(:,trial);
        end
    end

    results.observations.hints(block,:) = hint_outcomes;
    results.observations.rewards(block,:) = reward_outcomes;
    results.choices(:,:,block) = actions(:,:);
    results.R(:,block) = R(:,block);

    if block == 1
        results.input.task = task;
        results.input.params = params;
        results.input.observations = observations;
        results.input.choices = choices;
        results.input.sim = sim;
    end


    results.blockwise(block).action_probs = action_probs;
    results.blockwise(block).actions = actions;
    results.blockwise(block).true_context = true_context;
    results.blockwise(block).hint_outcomes = hint_outcomes;
    results.blockwise(block).hint_outcome_vector = hint_outcome_vector;
    results.blockwise(block).reward_outcomes = reward_outcomes;
    results.blockwise(block).action_values_Q = Q;
    results.blockwise(block).epistemic_value = epistemic_value;
    results.blockwise(block).pragmatic_value = pragmatic_value;
    results.blockwise(block).state_priors_d = dir_context;
    results.blockwise(block).norm_priors_d = p_context;
    results.blockwise(block).norm_posteriors_d_t2 = pp_context;
    results.blockwise(block).norm_posteriors_d_final = ppp_context;
    results.blockwise(block).trust_priors_a = a{1};
    results.blockwise(block).norm_trust_priors_a = A{1};
    results.blockwise(block).actualreward = actualreward;
end


% epistemic value term (Bayesian surprise) in expected free energy 
function G = G_epistemic_value(A,s)
    % A   - likelihood array (probability of outcomes given causes)
    % s   - probability density of causes
    % probability distribution over the hidden causes: i.e., Q(s)
    qx = spm_cross(s); % this is the outer product of the posterior over states
                    % calculated with respect to itself
    % accumulate expectation of entropy: i.e., E[lnP(o|s)]
    G     = 0;
    qo    = 0;
    for i = find(qx > exp(-16))'
        % probability over outcomes for this combination of causes
        po   = 1;
        for g = 1:numel(A)
            %po = spm_cross(po,A{g}(:,i));
            po = spm_cross(po,A(:,i));
        end
        po = po(:);
        qo = qo + qx(i)*po;
        G  = G  + qx(i)*po'*nat_log(po);
    end
    % subtract entropy of expectations: i.e., E[lnQ(o)]
    G  = G - qo'*nat_log(qo);
end 

function y = nat_log(x)
    y = log(x+exp(-16));
end 

function A  = spm_norm(A)
    % normalisation of a probability transition matrix (columns)
    %--------------------------------------------------------------------------
    A           = bsxfun(@rdivide,A,sum(A,1));
    A(isnan(A)) = 1/size(A,1);
end




