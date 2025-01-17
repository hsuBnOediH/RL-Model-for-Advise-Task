function result = advise_mf_uni(file)
    subdat = readtable(file);
    if strcmp(class(subdat.trial),'cell')
        subdat.trial = str2double(subdat.trial);
    end
    
    subdat = subdat(max(find(ismember(subdat.trial_type,'MAIN')))+1:end,:);
    % lets look at options selected
    left_right_chosen = subdat(subdat.event_type==8, :);
    % if the person managed to cause a glitch and select two bandits in one 
    % trial, use only the first one as response/result
    [~, idx] = unique(left_right_chosen.trial, 'first');
    left_right_chosen = left_right_chosen(idx, :);
    resp = left_right_chosen.response;

    result = subdat(subdat.event_type==9 & ~(strcmp(subdat.result,"try left")|strcmp(subdat.result,"try right")), :);
    points = result.result;
    % re = tp(~ismember(tp.result, {'try right', 'try left'}),:).result;
    % w_ad = tp(ismember(tp.result, {'try right', 'try left'}),{'trial' 'result'});

    got_advice = subdat.event_type ==9 & (strcmp(subdat.result,"try left")|strcmp(subdat.result,"try right"));
    trials_got_advice = subdat.trial(got_advice);
    advice_given = subdat.result(got_advice);
    trials_got_advice = trials_got_advice + 1;

    for n = 1:size(resp,1)
        % indicate if participant chose right or left
        if ismember(resp(n),'right')
            r=4;
        elseif ismember(resp(n),'left')
            r=3;
        elseif ismember(resp(n),'none')
            error("this person chose the did nothing option and our scripts are not set up to allow that")
        end 

        if str2double(points{n}) >0 
            pt=3;
        elseif str2double(points{n}) <0 
            pt=2;
        else
            error("this person chose the did nothing option and our scripts are not set up to allow that")
        end

        if ismember(n, trials_got_advice)
            u{n} = [1 2; 1 r]';
            index = find(trials_got_advice == n);
            if strcmp(advice_given{index}, 'try right')
                y = 3;
            elseif strcmp(advice_given{index}, 'try left')
                y = 2;
            end
            o{n} = [1 y 1; 1 1 pt; 1 2 r];
        else
            u{n} = [1 r; 1 1]';
            o{n} = [1 1 1; 1 pt 1; 1 r 1];
        end

    end
    % let's see how many times the person won, then for the next turn,
    % immediately chose other option
    win_trials = [];
    win_choose_other = [];
    % let's see how many times the person lost, then for the next turn,
    % immediately chose same option
    lose_trials = [];
    lose_choose_same = [];
    % let's see how many times the person followed bad advice from the advisor,
    % then chose advisor again
    advisor_led_astray = [];
    advisor_chosen_after_bad_advice = [];

    for m = 1:12
        for n = 1:30
            q = 30*(m-1);
            trial_number = n+q;
            if trial_number > 1 && trial_number <= length(o)
                curr_trial_obs = o(1,trial_number);
                curr_trial_obs = curr_trial_obs{:};
                prev_trial_obs = o(1,trial_number-1);
                prev_trial_obs = prev_trial_obs{:};
                advisor_prev_chosen = ismember(trial_number-1, trials_got_advice);
                if advisor_prev_chosen
                    previous_result = prev_trial_obs(2,3);
                    previous_choice = prev_trial_obs(3,3);
                    advisor_suggests_left = prev_trial_obs(1,2) == 2;
                    % if person followed advice
                    if ((advisor_suggests_left && previous_choice == 3) || (~advisor_suggests_left && previous_choice == 4))
                        % if person got it wrong
                        if previous_result == 2
                            advisor_led_astray = [advisor_led_astray trial_number-1];
                            if curr_trial_obs(1,2) ~= 1
                                advisor_chosen_after_bad_advice = [advisor_chosen_after_bad_advice trial_number];
                            end
                        end
                    end
                else
                    previous_result = prev_trial_obs(2,2);
                    previous_choice = prev_trial_obs(3,2);
                end
                % if win
                if previous_result == 3
                    win_trials = [win_trials trial_number-1];
                    % if chose other option immediately
                    if(~(curr_trial_obs(3,2) == previous_choice || curr_trial_obs(3,2) == 2))
                        win_choose_other = [win_choose_other trial_number];
                    end
                end
                % if lose 
                if previous_result == 2
                    lose_trials = [lose_trials trial_number-1];
                    % if chose same option immediately
                    if(curr_trial_obs(3,2) == previous_choice)
                        lose_choose_same = [lose_choose_same trial_number];
                    end
                end
            end
        end
    end

    percent_win_choose_other = length(win_choose_other)/length(win_trials);
    percent_lose_choose_same = length(lose_choose_same)/length(lose_trials);
    percent_choose_advisor_after_bad_advice = length(advisor_chosen_after_bad_advice)/length(advisor_led_astray);
    % first 6 blocks
    win_trials_first = win_trials(win_trials <= 180);
    win_choose_other_first = win_choose_other(win_choose_other <= 180);
    lose_trials_first = lose_trials(lose_trials <= 180);
    lose_choose_same_first = lose_choose_same(lose_choose_same <= 180);
    advisor_led_astray_first = advisor_led_astray(advisor_led_astray <= 180);
    advisor_chosen_after_bad_advice_first = advisor_chosen_after_bad_advice(advisor_chosen_after_bad_advice <= 180);

    percent_win_choose_other_first = length(win_choose_other_first)/length(win_trials_first);
    percent_lose_choose_same_first = length(lose_choose_same_first)/length(lose_trials_first);
    percent_choose_advisor_after_bad_advice_first = length(advisor_chosen_after_bad_advice_first)/length(advisor_led_astray_first);

    win_trials_second = win_trials(win_trials > 180);
    win_choose_other_second = win_choose_other(win_choose_other > 180);
    lose_trials_second = lose_trials(lose_trials > 180);
    lose_choose_same_second = lose_choose_same(lose_choose_same > 180);
    advisor_led_astray_second = advisor_led_astray(advisor_led_astray > 180);
    advisor_chosen_after_bad_advice_second = advisor_chosen_after_bad_advice(advisor_chosen_after_bad_advice > 180);

    percent_win_choose_other_second = length(win_choose_other_second)/length(win_trials_second);
    percent_lose_choose_same_second = length(lose_choose_same_second)/length(lose_trials_second);
    percent_choose_advisor_after_bad_advice_second = length(advisor_chosen_after_bad_advice_second)/length(advisor_led_astray_second);


    result = struct('percent_win_choose_other', percent_win_choose_other, ...
                        'percent_lose_choose_same', percent_lose_choose_same, ...
                        'percent_choose_advisor_after_bad_advice', percent_choose_advisor_after_bad_advice, ...
                        'time_advisor_chosen', length(trials_got_advice));
return


