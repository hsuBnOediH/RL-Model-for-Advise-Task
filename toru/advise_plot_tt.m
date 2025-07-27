function Q = advise_plot_cmg(whole_MDP)
% auxiliary plotting routine for spm_MDP_VB - multiple trials
% FORMAT Q = spm_MDP_VB_game(MDP)
%
% MDP.P(M,T)      - probability of emitting action 1,...,M at time 1,...,T
% MDP.Q(N,T)      - an array of conditional (posterior) expectations over
%                   N hidden states and time 1,...,T
% MDP.X           - and Bayesian model averages over policies
% MDP.R           - conditional expectations over policies
% MDP.O(O,T)      - a sparse matrix encoding outcomes at time 1,...,T
% MDP.S(N,T)      - a sparse matrix encoding states at time 1,...,T
% MDP.U(M,T)      - a sparse matrix encoding action at time 1,...,T
% MDP.W(1,T)      - posterior expectations of precision
%
% MDP.un  = un    - simulated neuronal encoding of hidden states
% MDP.xn  = Xn    - simulated neuronal encoding of policies
% MDP.wn  = wn    - simulated neuronal encoding of precision
% MDP.da  = dn    - simulated dopamine responses (deconvolved)
% MDP.rt  = rt    - simulated dopamine responses (deconvolved)
%
% returns summary of performance:
%
%     Q.X  = x    - expected hidden states
%     Q.R  = u    - final policy expectations
%     Q.S  = s    - initial hidden states
%     Q.O  = o    - final outcomes
%     Q.p  = p    - performance
%     Q.q  = q    - reaction times
%
% please see spm_MDP_VB
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_game.m 7307 2018-05-08 09:44:04Z karl $


total_trials = length(whole_MDP);  % Total number of trials
TrialsPerSubplot = 30;
NumSubplots = ceil(total_trials / TrialsPerSubplot);
F = spm_figure('GetWin', 'Graphics'); clf;
% Adjust the figure size
ScreenSize = get(0, 'ScreenSize');  % Get the screen size
FigureWidth = ScreenSize(3) * 0.7;  % e.g., 70% of the screen width
FigureHeight = ScreenSize(4) * 0.7;  % e.g., 70% of the screen height
Left = (ScreenSize(3) - FigureWidth) / 2;
Bottom = (ScreenSize(4) - FigureHeight) / 2;
F = spm_figure('GetWin', 'Graphics'); clf;
set(F, 'Position', [Left, Bottom, FigureWidth, FigureHeight]);
for subPlotIndx = 1:NumSubplots
    
    StartTrial = (subPlotIndx - 1) * TrialsPerSubplot + 1;
    EndTrial = (subPlotIndx * TrialsPerSubplot);
    MDP = whole_MDP(StartTrial:EndTrial);
    
    Rows = 6;  % Number of rows
    ColumnsPerRow = 2;  % Number of columns per row
    RowIndx = ceil(subPlotIndx / ColumnsPerRow);
    ColIndx = mod(subPlotIndx-1, ColumnsPerRow) + 1;
    SubplotIndx = (RowIndx - 1) * ColumnsPerRow + ColIndx;
    subplot(Rows, ColumnsPerRow, SubplotIndx);
    
    hold on;

    Nt = 30; % number of trials in a block
     for i = 1:Nt
        o(:,i) = MDP(i).o(2,:)';
        if isfield(MDP, 'P')
           act_prob(:,i) = MDP(i).P(:,:,1)'; % get action probs for first time step
        end
        act(:,i) = MDP(i).u(2,1);
        is_win_trial = any(o == 3, 1);
        
    end

    col   = {'r.','g.','b.','c.','m.','k.'};
    %t     = 1:Nt;
   % subplot(5,1,1)
    if Nt < 64
        MarkerSize = 24;
    else
        MarkerSize = 16;
    end
    if isfield(MDP, 'P')
       image(64*(1 - act_prob)),  hold on
    end

   % plot(act,col{3},'MarkerSize',MarkerSize)
   % plot win trials
   win_actions = nan(1,30);
   lose_actions = nan(1,30);
   for i = 1:length(act)
       if is_win_trial(i)
            win_actions(i) = act(i);
       end
   end
   % plot lose trials
   for i = 1:length(act)
       if ~is_win_trial(i)
            lose_actions(i) = act(i);
       end
   end
   
   plot(win_actions, 'g.', 'MarkerSize', MarkerSize); % Green dot if is_win_trial is 1
   plot(lose_actions, 'r.', 'MarkerSize', MarkerSize); % Red dot if is_win_trial is 0

    
    try
        plot(Np*(1 - act_prob(Np,:)),'r')
    end
    try
        E = spm_softmax(spm_cat({MDP.e}));
        plot(Np*(1 - E(end,:)),'r:')
    end
    
% Check if "reaction_times" exists in MDP
if isfield(MDP, 'reaction_times')
    % get max and mean reaction time for the block
    % Step 1: Extract reaction times into a cell array
    reaction_times_cells = {MDP.reaction_times};

    % Step 2: Concatenate all reaction times into a single array
    % Flatten the cell array and filter out NaN values
    all_reaction_times = cellfun(@(c) c(~isnan(c)), reaction_times_cells, 'UniformOutput', false);
    all_reaction_times = [all_reaction_times{:}];  % This creates a single array

    % Step 3: Calculate the mean and maximum, ignoring NaN values
    mean_reaction_time = mean(all_reaction_times, 'omitnan');
  %  mean_reaction_time = round(mean_reaction_time,2);
    max_reaction_time = max(all_reaction_times, [], 'omitnan');  % The '[]' ensures max operates on the entire array
   % max_reaction_time = round(max_reaction_time,2);
    % Now you have the mean and max without NaN values
end
    
    
    
    xlim([0 30]); 
    set(gca, 'YTick', 1:4, 'YTickLabel', {'Start','Hint          ', 'Left', 'Right          '});
if isfield(MDP, 'reaction_times')
    title(sprintf('Block %d;   Mean RT: %.2f, Max RT: %.2f', subPlotIndx, mean_reaction_time, max_reaction_time));
else
    title(sprintf('Block %d', subPlotIndx));
end
    %subtitle(sprintf('Mean RT: %d, Max RT: %d', mean_reaction_time, max_reaction_time));

    hold off;
end
disp("hi");