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
OUTPUT_PATH = getenv('OUTPUT_PATH');
output_folder_path = OUTPUT_PATH;



if isempty(F_TABLE_PATH_1)
    disp('F_TABLE_PATH_1 not set');
    return
end

if isempty(F_TABLE_PATH_2)
    disp('F_TABLE_PATH_2 not set');
    return
end

% read both csv files name
file_name_1 = F_TABLE_PATH_1.split('/').end();
file_name_2 = F_TABLE_PATH_2.split('/').end();

model_name_1 = file_name_1.split('-').get(0);
model_name_2 = file_name_2.split('-').get(0);

author_name_1 = file_name_1.split('-').get(1).split('_').get(0);
author_name_2 = file_name_2.split('-').get(1).split('_').get(0);

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

% get the input path, remove the csv extension add results to the name

output_file_name = strcat(model_name_1, '_', author_name_1, '_vs_', model_name_2, '_', author_name_2, '_bms.csv');


writetable(results_table, fullfile(output_folder_path, output_file_name));
