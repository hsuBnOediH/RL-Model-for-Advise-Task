function mdp = active_inference_model_mp_uni(task, MDP, params, sim)

    
    % priors that we are fixing
    params.p_right = .5; % right better prob in d
    %% !!! What is the that prior_d?
    params.prior_d = 1; % prior d counts
    params.prior_a = 1; % prior a counts
    params.learn_a = 1; % !!! a switch for learning???
    

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

    rs = params.Rsensitivity;

    p_a = params.p_a;
    % alpha = priors.alpha;
    eta_d_win = params.eta_d_win;
    eta_d_loss = params.eta_d_loss;
    eta_a_win = params.eta_a_win;
    eta_a_loss = params.eta_a_loss;
    % all the learning rate ---> eta, all the forgetting rate ---> omega
    omega_d_win = params.omega_d_win;
    omega_d_loss = params.omega_d_loss;
    omega_a_win = params.omega_a_win;
    omega_a_loss = params.omega_a_loss;
    novelty_scalar = 0;


    if strcmp(task.block_type,"SL")
        LA = 4;
    elseif strcmp(task.block_type,"LL")
        LA = 8;
    else
        print("block type not found")
    end
    
    for t = 1:size(task.true_p_right,2)
        % read the ground truth for each trail
        pHA = task.true_p_a(t);
        pLB = 1 - task.true_p_right(t);
        % use 4 and 4.7 for loss sizes of 4 and 8 because in exponentiated space
        % 4.7 is double 4
        % !!! should l_loss_value also be used in active_inference_model to replace this part? for fairness
        if (LA == 8)
            LA = log(exp(4)*2);
        end

        % num of time step
        T = 3;

        % !!! Should the ground truth plb be used in here for not simulating the data?
        % D ---> prior probabilities over initial states
        % Priors about initial states: D and d
        D{1} = [pLB 1-pLB]';  % {'left better','right better'}
        D{2} = [1 0 0 0]'; % {'start','hint','choose-left','choose-right'}

        d{1} = params.prior_d*[1-params.p_right params.p_right]';  % {'left better','right better'}
        d{2} = [1 0 0 0]'*200; % {'start','hint','choose-left','choose-right'}

        Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)

        % A ---> from states to outcomes, likelihood of outcomes given states
        % State-outcome mappings and beliefs: A and a

        % mapping from two states to first outcome modality hint
        % the first states factor is context represente by the left column and right column
        % the second states factor is choice states represented by the {i}
        % the row is the different outcome state of the hint outcome modality
        for i = 1:Ns(2) 
            A{1}(:,:,i) = [
                            1 1; % No Hint
                            0 0; % Machine-Left Hint
                            0 0 % Machine-Right Hint
                                ];
        end
        % Then we specify that the 'Get Hint' behavior state generates a hint that
        % either the left or right slot machine is better, depending on the context
        % state. In this case, the hints are accurate with a probability of pHA. 
        A{1}(:,:,2) = [
                        0     0;      % No Hint
                        pHA(1) 1-pHA(1);    % Machine-Left Hint
                        1-pHA(1) pHA(1)     % Machine-Right Hint
                                        ];  
        % Next we specify the mapping between states and wins/losses. The first two
        % behavior states ('Start' and 'Get Hint') do not generate either win or
        % loss observations in either context:

        % mapping from two states to second outcome modality win/lose
        % the first states factor is context represente by the left column and right column
        % the second states factor is choice states represented by the {i}
        % the row is the different outcome state of the win/lose outcome modality
        for i = 1:2
            A{2}(:,:,i) = [1 1;  % Null
                        0 0;  % Loss
                        0 0]; % Win
        end
                % Choosing the left machine (behavior state 3) generates wins with
        % probability pWin, which differs depending on the context state (columns):
        A{2}(:,:,3) = [0      0;     % Null        
                    0      1;  % Loss
                    1      0]; % Win

        % Choosing the right machine (behavior state 4) generates wins with
        % probability pWin, with the reverse mapping to context states from 
        % choosing the left machine:
                
        A{2}(:,:,4) = [0      0;     % Null
                    1      0;  % Loss
                    0      1]; % Win
                
        % Finally, we specify an identity mapping between behavior states and
        % observed behaviors, to ensure the agent knows that behaviors were carried
        % out as planned. Here, each row corresponds to each behavior state.

        % mapping from two states to third outcome modality observed action
        % the first states factor is context represente by the left column and right column
        % the second states factor is choice states represented by the {i}
        % the row is the different outcome state of the observed action outcome modality        
        for i = 1:Ns(2) 
            A{3}(i,:,i) = [1 1];
        end


        %!!! Why times 200?
        a{1} = A{1}*200;
        a{2} = A{2}*200;
        a{3} = A{3}*200;
                   
        a{1}(:,:,2) =  [0     0;      % No Hint
                    params.p_a 1-params.p_a;    % Machine-Left Hint
                    1-params.p_a params.p_a]*params.prior_a;   % Machine-Right Hint
        

        % B --> transitions among states 

        B{1}(:,:,1) = [1 0;  % 'Left Better' Context
                    0 1]; % 'Right Better' Context

        % Move to the Start state from any other state
        B{2}(:,:,1) = [1 1 1 1;  % Start State
                    0 0 0 0;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine
                
        % Move to the Hint state from any other state
        B{2}(:,:,2) = [0 0 0 0;  % Start State
                    1 1 1 1;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine

        % Move to the Choose Left state from any other state
        B{2}(:,:,3) = [0 0 0 0;  % Start State
                    0 0 0 0;  % Hint
                    1 1 1 1;  % Choose Left Machine
                    0 0 0 0]; % Choose Right Machine

        % Move to the Choose Right state from any other state
        B{2}(:,:,4) = [0 0 0 0;  % Start State
                    0 0 0 0;  % Hint
                    0 0 0 0;  % Choose Left Machine
                    1 1 1 1]; % Choose Right Machine    
                    
        % C -->(log) prior preferences for outcomes
        No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % number of outcomes in each outcome modality
        C{1}      = zeros(No(1),T); % Hints
        C{2}      = zeros(No(2),T); % Wins/Losses
        C{3}      = zeros(No(3),T); % Observed Behaviors

        % the row is the different outcome state of the win/lose outcome modality
        % the column is the different time step
        % only have preference for the win/lose outcome modality by pos or neg reward
        % !!! is the LA and rs and log(exp(4)/2)*rs need to be changed? to match other models?
        C{2}(:,:) =    [0      0        0;  % Null
                        0 -LA -LA  ;  % Loss
                        0      4*rs       log(exp(4)/2)*rs]; % win


        % For our simulations, we will specify V, where 
        % points and should be length T-1 (here, 2 transitions, from time point 1
        % to time point 2, and time point 2 to time point 3):
        Np = 4; % Number of policies
        Nf = 2; % Number of state factors


        % length T-1 (here, 2 transitions, from time point 1 to time point 2, and time point 2 to time point 3):
        % rows correspond to time 
        V = ones(T-1,Np,Nf);

        % note that I took out the policy where an agent can take the hint (2) and
        % then do nothing (1) as well as do nothing and do nothing 
        % the row is time steps
        % the column is the different policies
        % the first slice is for context feature states, the second slice is for choice feature states
        % all the values are 1, since the context state is not controllable
        V(:,:,1) = [1 1 1 1;
                    1 1 1 1];
        

        V(:,:,2) = [2 2 3 4;
                    3 4 1 1];
        
        % For V(:,:,2), columns left to right indicate policies allowing: 
        % 1. staying in the start state 
        % 2. choosing the hint then returning to start state
        % 3. taking the hint then choosing the left machine
        % 4. taking the hint then choosing the right machine
        % 5. choosing the left machine right away (then returning to start state)
        % 6. choosing the right machine right away (then returning to start state)


        
        beta = 1; % By default this is set to 1, but try increasing its value 
        % to lower precision and see how it affects model behavior

        erp = 1; % By default we here set this to 1, but try increasing its value  
            % to see how it affects simulated neural (and behavioral) responses
            
        % changed tau to 2; quicker jumps                          
        tau = 2; % Here we set this to 12 to simulate smooth physiological responses,   
            % but try adjusting its value to see how it affects simulated
                % neural (and behavioral) responses



        mdp(t).T = T;                    % Number of time steps
        mdp(t).V = V;                    % allowable (deep) policies

        %mdp.U = U;                   % We could have instead used shallow 
                                    % policies (specifying U instead of V).

        mdp(t).A = A;                    % state-outcome mapping
        mdp(t).B = B;                    % transition probabilities
        mdp(t).C = C;                    % preferred states
        mdp(t).D = D;                    % priors over initial states

        mdp(t).d = d;                    % enable learning priors over initial states



        mdp(t).eta_a_win = eta_a_win;
        mdp(t).eta_a_loss = eta_a_loss;
        mdp(t).eta_d_win = eta_d_win;
        mdp(t).eta_d_loss = eta_d_loss;
        
        mdp(t).omega_a_win = omega_a_win;
        mdp(t).omega_a_loss = omega_a_loss;
        mdp(t).omega_d_win = omega_d_win;
        mdp(t).omega_d_loss = omega_d_loss;

        % mdp(t).alpha = alpha;            % action precision fixed at 1
        mdp(t).beta = beta;              % expected precision of expected free energy over policies
        mdp(t).erp = erp;                % degree of belief resetting at each timestep
        mdp(t).tau = tau;                % time constant for evidence accumulation
        mdp(t).prior_d = params.prior_d;
        mdp(t).p_a = params.p_a;
        mdp(t).prior_a = params.prior_a;
        mdp(t).rs = rs;
        mdp(t).novelty_scalar = novelty_scalar;

        % if learn_a ==1
        %     mdp(t).a = a;   
        % end

        %!!! What is the a_floor and d_floor?, is the d_floor same as context_floor?
        mdp(t).a_floor = a{1}(:,:,2);
        mdp(t).d_floor = d{1};
        %% Label of all the states and actions
        %--------------------------------------------------------------------------
        % states features 1: context
        label.factor{1}   = 'contexts';   
        label.name{1}    = {'left-better','right-better'};
        % states features 2: choice states
        label.factor{2}   = 'choice states';     
        label.name{2}    = {'start','hint','choose left','choose right'};
        
        % outcome modality 1: hint
        label.modality{1} = 'hint';    
        label.outcome{1} = {'null','left hint','right hint'};
        % outcome modality 2: win/lose
        label.modality{2} = 'win/lose';  
        label.outcome{2} = {'null','lose','win'};
        % outcome modality 3: observed action
        label.modality{3} = 'observed action';  
        label.outcome{3} = {'start','hint','choose left','choose right'};

        % actions
        % !!! Why the label.action starts with 2?
        label.action{2} = {'start','hint','left','right'};

        mdp(t).label = label;
    end

    

    %--------------------------------------------------------------------------
    % Use a script to check if all matrix-dimensions are correct:
    %--------------------------------------------------------------------------
    mdp = spm_MDP_check(mdp);

    mdp = spm_MDP_VB_X_advice(mdp);



end