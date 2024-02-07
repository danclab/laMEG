function [source_ts, time]=load_source_time_series(data_D, inv_D, vertices, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

spm_get_defaults('use_parfor',1);
spm_get_defaults('mat.format','-v7.3');

D_data=spm_eeg_load(data_D);
time = D_data.time;
D_inv=spm_eeg_load(inv_D);

if length(vertices)>0
    M=D_inv.inv{1}.inverse.M(vertices,:);
else
    M=D_inv.inv{1}.inverse.M;
end
U=D_inv.inv{1}.inverse.U{1};
MU=M*U;

megchans=D_data.indchantype('meg','good');

source_ts = zeros(size(MU, 1), size(D_data, 2), size(D_data, 3));

% Loop over trials
for trial = 1:size(D_data, 3)
    % Perform matrix multiplication for each trial
    source_ts(:, :, trial) = MU * D_data(megchans, :, trial);
end
source_ts=single(source_ts);