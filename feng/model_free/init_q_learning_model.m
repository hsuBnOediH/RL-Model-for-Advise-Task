function model = init_q_learning_model(params)
    % Define states and actions
    states = {'start', 'win', 'loss', 'advise_win', 'advise_loss', 'advise_left', 'advise_right'};
    actions = {'left', 'right', 'take_advise'};
    
    % Initialize Q-table with zeros, and set NaNs for unreachable pairs as needed
    q_table = zeros(length(states), length(actions));
    q_table(strcmp(states, 'start'), strcmp(actions, 'take_advise')) = NaN;
    q_table(strcmp(states, 'win'), :) = NaN;
    q_table(strcmp(states, 'loss'), :) = NaN;
    
    % Store states, actions, and Q-table in the model structure
    model.states = states;
    model.actions = actions;
    model.q_table = q_table;
    
    % Set learning rates based on conditions
    if isfield(params, 'lr')
        % General learning rate provided, set all learning rates to this value
        model.lr_a_win = params.lr;
        model.lr_a_loss = params.lr;
        model.lr_d_win = params.lr;
        model.lr_d_loss = params.lr;
        
    else
        % Specific learning rates provided
        if isfield(params, 'lr_a')
            model.lr_a_win = params.lr_a;
            model.lr_a_loss = params.lr_a;
        else
            % Set individually or default to zero
            if isfield(params, 'lr_a_win')
                model.lr_a_win = params.lr_a_win;
            else
                model.lr_a_win = 0;
            end
            if isfield(params, 'lr_a_loss')
                model.lr_a_loss = params.lr_a_loss;
            else
                model.lr_a_loss = 0;
            end
        end
        
        if isfield(params, 'lr_d')
            model.lr_d_win = params.lr_d;
            model.lr_d_loss = params.lr_d;
        else
            % Set individually or default to zero
            if isfield(params, 'lr_d_win')
                model.lr_d_win = params.lr_d_win;
            else
                model.lr_d_win = 0;
            end
            if isfield(params, 'lr_d_loss')
                model.lr_d_loss = params.lr_d_loss;
            else
                model.lr_d_loss = 0;
            end
        end
    end
    
    % Set remaining parameters (discount factor and inverse temperature)
    if isfield(params, 'discount_factor')
        model.discount_factor = params.discount_factor;
    else
        model.discount_factor = 0.9; % Default to 0.9 if not provided
    end
    
    if isfield(params, 'inv_temp')
        model.inv_temp = params.inv_temp;
    else
        model.inv_temp = 1.0; % Default to 1.0 if not provided
    end
end