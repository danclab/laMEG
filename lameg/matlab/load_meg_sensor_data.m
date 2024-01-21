function sensor_data=load_meg_sensor_data(data_D, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

D_data=spm_eeg_load(data_D);

megchans=D_data.indchantype('meg','good');

sensor_data = D_data(megchans, :, :);