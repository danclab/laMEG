function [f_vals,wois]=invert_sliding_window(prior, data_file,...
    patch_size, n_temp_modes, win_size,...
    win_overlap, foi, hann, gain_mat_fname, viz, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

spm_get_defaults('cmdline',~viz);
spm_get_defaults('use_parfor',1);
spm_get_defaults('mat.format','-v7.3');

% Data file to load
D=spm_eeg_load(data_file);

% Create wois
times=D.time*1000;
dt=times(2)-times(1);
win_steps=round(win_size/dt);
wois=[];
if win_overlap
    for t_idx=1:length(times)
        win_l=max(1,ceil(t_idx-win_steps/2));
        win_r=min(length(times),floor(t_idx+win_steps/2));
        woi=[times(win_l) times(win_r)];
        wois(t_idx,:)=woi;
    end
else
    ts=linspace(times(1),times(end),(times(end)-times(1))./win_size);
    wois=[];
    for i=2:length(ts)
        wois(end+1,:)=[ts(i-1) ts(i)];
    end
end

% Setup spatial modes for cross validation
[data_dir,fname,ext]=fileparts(data_file);
spatialmodesname=fullfile(data_dir, sprintf('%s_testmodes.mat',fname));
[spatialmodesname,Nmodes,pctest]=spm_eeg_inv_prep_modes_xval(data_file, [], spatialmodesname, 1, 0);

% so use all vertices that will be simulated on (plus a few more) as MSP priors
Ip=[prior];
% Save priors
patchfilename=fullfile(data_dir, 'patch.mat');
save(patchfilename,'Ip');

clear jobs
matlabbatch={};
batch_idx=1;

% Source reconstruction
matlabbatch{batch_idx}.spm.meeg.source.invertiter.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.whatconditions.all = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invfunc = 'Classic';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invtype = 'MSP'; %;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.woi = wois;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.foi = foi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.hanning = hann;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.fixedpatch.fixedfile = {patchfilename}; % '<UNDEFINED>';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.fixedpatch.fixedrows = [1 Inf]; %'<UNDEFINED>';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.patchfwhm = -patch_size; %% NB A fiddle here- need to properly quantify
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.mselect = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.nsmodes = Nmodes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.umodes = {spatialmodesname};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.ntmodes = n_temp_modes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.priorsmask = {''};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.space = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.radius = 32;
if length(gain_mat_fname)
    matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.gain_mat = {gain_mat_fname};
end
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.outinv = '';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.modality = {'All'};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.crossval = [pctest 1];


[a,~]=spm_jobman('run', matlabbatch);
% Get F-values for inversion
Drecon=spm_eeg_load(a{1}.D{1});                
f_vals=Drecon.inv{1}.inverse.crossF;
