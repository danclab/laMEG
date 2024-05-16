function source_power=load_source_power(data_D, inv_D, woi, vertices, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

spm_get_defaults('use_parfor',1);
spm_get_defaults('mat.format','-v7.3');

D_data=spm_eeg_load(data_D);
time = D_data.time;
t_idx=1:length(time);
if length(woi)>0
    t_idx=find((time>=woi(1) && (time<=woi(2))));
end
D_inv=spm_eeg_load(inv_D);

if length(vertices)>0
    M=D_inv.inv{1}.inverse.M(vertices,:);
else
    M=D_inv.inv{1}.inverse.M;
end
U=D_inv.inv{1}.inverse.U{1};
MU=M*U;

megchans=D_data.indchantype('meg','good');

source_power = zeros(size(MU, 1), size(D_data, 3));

% Loop over trials
for trial = 1:size(D_data, 3)
    % Perform matrix multiplication for each trial
    source_power(:, trial) = var(MU * D_data(megchans, t_idx, trial),0,2);
end
source_power=single(source_power);