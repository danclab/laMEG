function source_ts=load_source_time_series(data_D, inv_D, vertices, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

D_data=spm_eeg_load(data_D);
D_inv=spm_eeg_load(inv_D);

if length(vertices)>0
    M=D_inv.inv{1}.inverse.M(vertices,:);
else
    M=D_inv.inv{1}.inverse.M;
end
U=D_inv.inv{1}.inverse.U{1};
MU=M*U;

megchans=D_data.indchantype('meg','good');

source_ts=MU*D_data(megchans,:,:);