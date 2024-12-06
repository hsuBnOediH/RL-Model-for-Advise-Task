function model = init_q_learning_model(params,all_params)
    % Define states and actions
    states = {'start', 'advise_left', 'advise_right'};
    actions = {'left', 'right', 'take_advise'};
    % Create containers.Map to map strings to indices
    state_map = containers.Map(states, 1:length(states));
    action_map = containers.Map(actions, 1:length(actions));

    left_better = all_params.left_better;
    advise_truthness = all_params.advise_truthness;

    % Initialize Q-table with zeros, and set NaNs for unreachable pairs as needed
    q_table = zeros(length(states), length(actions));
    % q_table(1, 1) = 40 * left_better;
    % q_table(1, 2) = 40 * (1-left_better);
    % q_table(1, 3) = 20 * advise_truthness;
    % q_table(2, 1) = 20 * left_better;
    % q_table(2, 2) = 20 * (1-left_better);
    % q_table(2, 3) = NaN;
    % q_table(3, 1) = 20 * left_better;
    % q_table(3, 2) = 20 * (1-left_better);
    % q_table(3, 3) = NaN;


    % q_table(1,1) = 40* left_better +(-60)* (1-left_better);
    % q_table(1,2) = 40* (1-left_better) +(-60)* left_better;
    % q_table(1,3) = 20* advise_truthness +(-60)* (1-advise_truthness);
    % q_table(2,1) = 20* left_better +(-60)* (1-left_better);
    % q_table(2,2) = 20* (1-left_better) +(-60)* left_better;
    % q_table(2,3) = NaN;
    % q_table(3,1) = 20* left_better +(-60)* (1-left_better);
    % q_table(3,2) = 20* (1-left_better) +(-60)* left_better;
    % q_table(3,3) = NaN;
    
    % Store states, actions, and Q-table in the model structure
    model.states = states;
    model.actions = actions;
    model.q_table = q_table;
    model.params = params;
    model.fields= fieldnames(params);
    prior_variance = .5;

    for i = 1:length(model.fields)
        model.con_var{i,i}    = prior_variance;
    end
    model.con_var = diag(cell2mat(model.con_var));
end