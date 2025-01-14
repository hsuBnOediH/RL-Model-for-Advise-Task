function model = init_q_learning_model(params,zero_one_fields,positive_fields)
    % Define states and actions
    states = {'start', 'advise_left', 'advise_right'};
    actions = {'left', 'right', 'take_advise'};
    % Create containers.Map to map strings to indices


    % Initialize Q-table with zeros, and set NaNs for unreachable pairs as needed
    q_table = zeros(length(states), length(actions));
    
    % Store states, actions, and Q-table in the model structure
    model.states = states;
    model.actions = actions;
    model.q_table = q_table;
    model.params = params;
    model.fields= fieldnames(params);
    prior_variance = .5;

    for i = 1:length(model.fields)
        field = model.fields{i};
        if ismember(field, zero_one_fields)
            model.params.(field) = log(model.params.(field) / (1 - model.params.(field)));
        elseif ismember(field, positive_fields)
            model.params.(field) = log(model.params.(field));
        end
        model.con_var{i,i}    = prior_variance;
    end
    model.con_var = spm_cat(model.con_var);
end