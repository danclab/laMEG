function [sensor_data, time, ch_names]=load_meg_sensor_data(data_D, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

spm_get_defaults('use_parfor',1);
spm_get_defaults('mat.format','-v7.3');

D_data=spm_eeg_load(data_D);

megchans=D_data.indchantype('meg','good');

sensor_data = D_data(megchans, :, :);
ch_names = D_data.chanlabels(megchans);
time = D_data.time;