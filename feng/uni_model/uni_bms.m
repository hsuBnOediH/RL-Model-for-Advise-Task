ROOT = ''; 
if isempty(ROOT)
    ROOT = fileparts(mfilename('fullpath'));
    disp(['ROOT path set to: ', ROOT]);
end

% Detect the system
% 'pc' for Windows, 'mac' for local Mac, 'cluster' for running on VM cluster
env_sys = '';
if ispc
    env_sys = 'pc';
elseif ismac
    env_sys = 'mac';
elseif isunix
    env_sys = 'cluster';
else
    disp('Unknown operating system.');
end
% Add external paths depending on the system
if strcmp(env_sys, 'pc')
    spmPath = '/output.csv';
elseif strcmp(env_sys, 'mac')
    spmPath =  [ ROOT '/../spm/'];
elseif strcmp(env_sys, 'cluster')
    spmPath = '/mnt/dell_storage/labs/rsmith/all-studies/util/spm12';

end
addpath(spmPath);


F_TABLE_PATH_1 = getenv('F_TABLE_PATH_1');
F_TABLE_PATH_2 = getenv('F_TABLE_PATH_2');
MODLE_NAME_1 = getenv('MODLE_NAME_1');
MODLE_NAME_2 = getenv('MODLE_NAME_2');
AUTHOR_NAME_1 = getenv('AUTHOR_NAME_1');
AUTHOR_NAME_2 = getenv('AUTHOR_NAME_2');
OUTPUT_PATH = getenv('OUTPUT_PATH');
output_folder_path = OUTPUT_PATH;
disp([]);
disp(['F_TABLE_PATH_1: ', F_TABLE_PATH_1]);
disp(['F_TABLE_PATH_2: ', F_TABLE_PATH_2]);
disp(['OUTPUT_PATH: ', OUTPUT_PATH]);

if isempty(F_TABLE_PATH_1)
    disp('F_TABLE_PATH_1 not set');
    return
end

if isempty(F_TABLE_PATH_2)
    disp('F_TABLE_PATH_2 not set');
    return
end


% read both csv files
allData_1 = readtable(F_TABLE_PATH_1);
allData_2 = readtable(F_TABLE_PATH_2);

% concatenate both tables, column wise, excluding the first column of table 2
allData = [allData_1, allData_2(:,2:end)];
% exclude the first column and first row
selectedColumns = allData(2:end, 2:end);
lme = table2array(selectedColumns);
Nsamp = size(lme,1);

[alpha,exp_r,xp,pxp,bor] = spm_BMS(lme, Nsamp,1);
results_table = table;
results_table.model = (1:size(lme,2))';
results_table.alpha = alpha';
results_table.exp_r = exp_r';
results_table.xp = xp';
results_table.pxp = pxp';
results_table.bor = repmat(bor,size(lme,2),1);
disp(results_table);

% get the input path, remove the csv extension add results to the name
disp(["output_folder_path: ", output_folder_path]);
output_file_name = strcat('bms_results-', MODLE_NAME_1, '_', AUTHOR_NAME_1, '-', MODLE_NAME_2, '_', AUTHOR_NAME_2, '.csv');
disp(['output_file_name: ', output_file_name]);

writetable(results_table, fullfile(output_folder_path, output_file_name));
