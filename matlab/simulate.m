function sim_file=simulate(data_file, prefix, sim_vertices, sim_woi,...
    sim_signals, dipole_orientations, dipole_moments, sim_patch_sizes, SNR,...
    average, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

Dmesh=spm_eeg_load(data_file);

[Dnew,~]=spm_eeg_simulate({Dmesh}, prefix, sim_vertices, sim_signals,...
    dipole_orientations, sim_woi, [], SNR,...
    [], [], sim_patch_sizes, dipole_moments, []);

[a1, ~, ~]=fileparts(data_file);
sim_file=fullfile(a1, Dnew.fname);

if average
    clear jobs
    matlabbatch={};
    batch_idx=1;

    matlabbatch{batch_idx}.spm.meeg.averaging.average.D = {sim_file};
    matlabbatch{batch_idx}.spm.meeg.averaging.average.userobust.standard = false;
    matlabbatch{batch_idx}.spm.meeg.averaging.average.plv = false;
    matlabbatch{batch_idx}.spm.meeg.averaging.average.prefix = 'm';
    spm_jobman('run', matlabbatch);
    sim_file=fullfile(a1, sprintf('m%s',Dnew.fname));
end