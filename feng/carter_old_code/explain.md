#### Input
- MDP
- OPTIONS


```Matlab
try OPTIONS.plot;  catch, OPTIONS.plot  = 0; end
% note that default now is without gamma
try OPTIONS.gamma; catch, OPTIONS.gamma = 1; end
try OPTIONS.D;     catch, OPTIONS.D     = 0; end
```
This snippet of code set defult values for the options. The default value for plot is 0, gamma is 1, and D is 0.


```Matlab
MDP = spm_MDP_check(MDP);
```
This line of code checks the MDP structure to make sure it is in the correct format. If it is not, it will throw an error.


```Matlab
if size(MDP,2) > 1
    
    % plotting options
    %----------------------------------------------------------------------
    GRAPH        = OPTIONS.plot;
    OPTIONS.plot = 0;
    
    for i = 1:size(MDP,2)
        for m = 1:size(MDP,1)

```

This Section only runs if there are multiple trail for each MDP model
The first dim of MDP is the number of models, and the second dim is the number of trials.

Graph is set to the value of OPTIONS.plot, and then OPTIONS.plot is set to 0 for preventing further plotting during the loop and to only plot at the end if necessary

Then it will loop all the model and trials, m as model index, and i as trial index

```Matlab
            
            % update concentration parameters
            % as in cary them over from the last trial
            %--------------------------------------------------------------
            if i > 1
                try,  MDP(m,i).a = OUT(m,i - 1).a; end
                try,  MDP(m,i).b = OUT(m,i - 1).b; end
                try,  MDP(m,i).d = OUT(m,i - 1).d; end
                try,  MDP(m,i).c = OUT(m,i - 1).c; end
                try,  MDP(m,i).e = OUT(m,i - 1).e; end
                
                % update initial states (post-diction)
                %----------------------------------------------------------
                if OPTIONS.D
                    for f = 1:numel(MDP(m,i).D)
                        MDP(m,i).D{f} = OUT(m,i - 1).X{f}(:,1);
                    end
                end
            end
        end
```
For each model m in trail in, the concetrantion pratermeters updated from the previous trial will be carried over.

if the flag D is set (the inite states need to be updated),the code will upadte the D also
where f is the number of factors in the model, and the initial states are updated from the previous trial.
OUT(m,i - 1).X{f}. It means that the code is selecting the first time step (or the first entry) of the f-th factor's hidden state from the previous trial (i - 1).

```Matlab
        % inference
        %----------------------------------------------------------------------
        OUT(m,i) = spm_MDP_VB_X(MDP(m,i),OPTIONS);
        
```
called to perform the actual variational message passing and inference for the current trial i.
The result of this inference is stored in OUT(:,i).

```Matlab
        if isfield(OPTIONS,'BMR')
            for m = 1:size(MDP,1)
                OUT(m,i) = spm_MDP_VB_sleep(OUT(m,i),OPTIONS.BMR);
            end
        end
```
if BMR is set in the options, the code will call spm_MDP_VB_sleep to perform the Bayesian model reduction.
this function performs model reduction (simplifying or reducing the complexity of the model) and updates the model output accordingly.

```Matlab
        MDP = OUT;
        % plot
        %----------------------------------------------------------------------
        if GRAPH
            spm_figure('GetWin','MDP');
            spm_MDP_VB_trial(MDP(m,i),OUT(m,i),OPTIONS);
        end
    end
```

The code will update the MDP structure with the results of the last trial, and if GRAPH is set, it will plot the results of the last trial.

```Matlab
    % set up and preliminaries
    %==========================================================================

    % defaults
    %--------------------------------------------------------------------------
    try, alpha = MDP(1).alpha; catch, alpha = 512;  end
    try, beta  = MDP(1).beta;  catch, beta  = 1;    end
    try, zeta  = MDP(1).zeta;  catch, zeta  = 3;    end
    try, eta_win   = MDP(1).eta_win;   catch, eta_win   = 1;    end
    try, eta_loss   = MDP(1).eta_loss;   catch, eta_loss   = 1;    end
    try, eta   = MDP(1).eta;   catch, eta   = 1;    end
    try, tau   = MDP(1).tau;   catch, tau   = 4;    end
    try, chi   = MDP(1).chi;   catch, chi   = 1/64; end
    try, erp   = MDP(1).erp;   catch, erp   = 4;    end
    try, p_ha  = MDP(1).p_ha;  catch, p_ha  = .75;    end
    try, rs    = MDP(1).rs;    catch, rs    = .4;    end
    try, omega = MDP(1).omega; catch, omega = 1;    end % forgetting rate
    try, prior_a = MDP(1).prior_a; catch, prior_a = 1;end
    try, prior_d = MDP(1).prior_d; catch, prior_d = 1;end
    try, omega_a = MDP(1).omega_a; catch, omega_a = 1;end
    try, omega_d = MDP(1).omega_d; catch, omega_d = 1;end
    %try, la = MDP(1).la;           catch, la = .1;end
    %try, eff = MDP(1).eff;         catch, eff = 1;end
    % preclude precision updates for moving policies
    %--------------------------------------------------------------------------
    if isfield(MDP,'U'), OPTIONS.gamma = 1;         end
```
this section of code are try to ensure the paramss are assigned safely, 
even some are missing in the MDP structure, the code will assign the default value to them.

the following table list the variables and their default values, meaning 

| Variable | Default Value |Meaning |
| --- | --- | --- |
| alpha | 512 | precision for action selection |
| beta | 1 | precision over precision |
| zeta | 3 | Occam's window for policy |
| eta_win | 1 | learning rate for win |
| eta_loss | 1 | learning rate for loss |
| eta | 1 | learning rate for both win and loss |
| tau | 4 | time constant for step size |
| chi | 1/64 | a threshold to control the stopping of the search |
| erp | 4 | time constant for leanring |
| p_ha | .75 | prior knowledge or bias towards certain outcomes in the model. |
| rs | .4 | a parameter controlling certain updates in the model |
| omega | 1 | forgetting rate |
| prior_a | 1 |  prior distributions for certain model parameters, helping to regularize the model and control the influence of prior beliefs. |
| prior_d | 1 |  prior distributions for certain model parameters, helping to regularize the model and control the influence of prior beliefs. |
| omega_a | 1 |  forgetting rate under the advice|
| omega_d | 1 |  forgetting rate under certain conditions |
|-|-|-|

if MDP.U exist in MMDP, the gamma will be 1 to enable specific behavior


```Matlab

    for m = 1:size(MDP,1)

        if isfield(MDP(m),'O') && size(MDP(m).U,2) < 2

            % no policies – assume hidden Markov model (HMM)
            %------------------------------------------------------------------
            T(m) = size(MDP(m).O{1},2);         % HMM mode
            V{m} = ones(T - 1,1);               % single 'policy'
            HMM  = 1;

        elseif isfield(MDP(m),'U')

            % called with repeatable actions (U,T)
            %------------------------------------------------------------------
            T(m) = MDP(m).T;                    % number of updates
            V{m} = MDP(m).U;                    % allowable actions (1,Np)
            HMM  = 0;

        elseif isfield(MDP(m),'V')

            % full sequential policies (V)
            %------------------------------------------------------------------
            V{m} = MDP(m).V;                    % allowable policies (T - 1,Np)
            T(m) = size(MDP(m).V,1) + 1;        % number of transitions
            HMM  = 0;

        else
            sprintf('Please specify MDP(%d).U, MDP(%i).V or MDP(%d).O',m), return
        end

    end
```
this code lopp all the models in MDP
HMM Mode: If the model has outcomes (O) but no repeatable actions or policies (U with fewer than 2 columns), it is assumed to be a Hidden Markov Model (HMM) with no specific action policies. The number of time steps (T(m)) is set based on the number of outcomes, and a single policy is assigned.



Repeatable Actions Mode: If the model has repeatable actions (U), the number of time steps (T(m)) is set to the value in MDP(m).T, and the allowable actions (V{m}) are set to the actions defined in U.

Full Sequential Policies Mode: If the model has full sequential policies (V), the number of time steps (T(m)) is set to the number of rows in V plus one (since transitions are one less than time steps), and the sequential policies are used.

other case will cause error



#### SET UP
This section of code is for inite MDP and essential paramser for running the VMP

```Matlab
    % initialise model-specific variables
    %--------------------------------------------------------------------------
    T     = T(1);                              % number of time steps
    Ni    = 8;                                % number of VB iterations
```

the number of time strap sis based on the firest model
Ni: number of VB iterations

then it will loop through all the models

first of all, it ensure the policy length is less than the number of updates
if the numner policy V{m} is exceed the number of time steps, it will be truncated to the number of time steps minus 1
the ":,:" means the truncation is applied to all the rows and columns of the policy matrix
```Matlab
    if size(V{m},1) > (T - 1)
        V{m} = V{m}(1:(T - 1),:,:);
    end
```

then the number of outcome factors, the number of hidden state factors and number of allowable police will be asssign
numel means the number of elements in the array
MDP(m).A is the outcome factors, MDP(m).B is the hidden state factors, and V{m} is the allowable policies
```Matlab
    Ng(m) = numel(MDP(m).A);               % number of outcome factors
    Nf(m) = numel(MDP(m).B);               % number of hidden state factors
    Np(m) = size(V{m},2);                  % number of allowable policies
```


#### This belief Updating Section for HS and Outcomes

the outside loop is each time step, the inner loop for each model
```Matlab
    for t = 1:T
        for m = M(t,:)
```

Then for a section for the non HMM Model only, in this section
all the factor will be looped, For each hidden factor f in model m, if the state at time t is not specified, it will be generated based on the previous state and action or initial state.
depending on the if t ==1, the initial state will be generated from the prior distribution or the previous state and action

```Matlab
    for f = 1:Nf(m)
        
        % the next state is generated by action
        %----------------------------------------------------------
        if MDP(m).s(f,t) == 0
            if t > 1
                ps = MDP(m).B{f}(:,MDP(m).s(f,t - 1),MDP(m).u(f,t - 1));
            else
                ps = spm_norm(MDP(m).D{f});
            end
            MDP(m).s(f,t) = find(rand < cumsum(ps),1);
        end
        
    end
```

then the posterior will processed in a pretty simliary way
For each hidden factor f, compute the predicted values for the next time step based on the previous state (xqq) and the Bayesian model average (xq),
depends on if it is firest time step

```Matlab
    for f = 1:Nf(m)
        if t > 1
            xqq{m,f} = sB{m,f}(:,:,MDP(m).u(f,t - 1)) * X{m,f}(:,t - 1);
        else
            xqq{m,f} = X{m,f}(:,t);
        end
        xq{m,f} = X{m,f}(:,t);
    end
```


If the outcome is generated by model n, the probability po is computed based on the Bayesian model average (xqq(m,:)), and the outcome is selected using the softmax function.
If the outcome comes from another model n, the outcome is copied from the corresponding model n.
If the outcome is explicitly specified as 0, a new outcome is sampled from the generative model A{g}.

```Matlab

            % sample outcome, if not specified
            %--------------------------------------------------------------
            for g = 1:Ng(m)
                
                if MDP(m).o(g,t) < 0
                    
                    % outcome is generated by model n
                    %------------------------------------------------------
                    n = -MDP(m).o(g,t);
                    MDP(m).n(g,t) = n;
                    if n == m
                        
                        % outcome that minimises expected free energy
                        %--------------------------------------------------
                        po    = spm_dot(A{m,g},xqq(m,:));
                        px    = spm_vec(spm_cross(xqq(m,:)));
                        F     = zeros(No(m,g),1);
                        for i = 1:No(m,g)
                            xp   = MDP(m).A{g}(i,:);
                            xp   = spm_norm(spm_vec(xp));
                            F(i) = spm_vec(px)'*spm_log(xp) + spm_log(po(i));
                        end
                        po            = spm_softmax(F*512);
                        MDP(m).o(g,t) = find(rand < cumsum(po),1);
                        
                    else
                        
                        % outcome from model n
                        %--------------------------------------------------
                        MDP(m).o(g,t) = MDP(n).o(g,t);
                        
                    end
                    
                elseif MDP(m).o(g,t) == 0
                    
                    % sample outcome from the generative process
                    %------------------------------------------------------
                    ind           = num2cell(MDP(m).s(:,t));
                    po            = MDP(m).A{g}(:,ind{:});
                    MDP(m).o(g,t) = find(rand < cumsum(po),1);
                    
                end
            end
```


