function model = init_model_based_model(params)
    % Define states and actions
    states = {'start','advise_left', 'advise_right'};
    actions = {'left', 'right', 'take_advise'};
    % Create containers.Map to map strings to indices
    state_map = containers.Map(states, 1:length(states));
    action_map = containers.Map(actions, 1:length(actions));

    % Initialize prob table with 0.5 as the prior
    prob_table = 0.5 * ones(3, 3);
    
    % Store states, actions, and Q-table in the model structure
    model.states = states;
    model.actions = actions;
    model.prob_table = prob_table;
    model.params = params;
    model.fields= fieldnames(params);
    prior_variance = .5;

    for i = 1:length(model.fields)
        model.con_var{i,i}    = prior_variance;
    end
    model.con_var = diag(cell2mat(model.con_var));
end