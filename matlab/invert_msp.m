function varargout=invert_msp(data_file, priors, patch_size, n_temp_modes, foi, woi, Nfolds,...
    ideal_pctest, gain_mat_fname, viz, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

spm_get_defaults('cmdline',~viz);

% Setup spatial modes for cross validation
[data_dir,fname,~]=fileparts(data_file);
spatialmodesname=fullfile(data_dir, sprintf('%s_testmodes.mat',fname));
[spatialmodesname,Nmodes,pctest]=spm_eeg_inv_prep_modes_xval(data_file, [], spatialmodesname, Nfolds, ideal_pctest);

if length(priors)>0
    Ip=priors;
    % Save priors
    patchfilename=fullfile(data_dir, 'patch.mat');
    save(patchfilename,'Ip');
end

clear jobs
matlabbatch={};
batch_idx=1;

% Source reconstruction
matlabbatch{batch_idx}.spm.meeg.source.invertiter.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.whatconditions.all = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invfunc = 'Classic';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invtype = 'MSP'; %;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.woi = woi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.foi = foi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.hanning = 1;
if length(priors)>0
    matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.fixedpatch.fixedfile = {patchfilename}; % '<UNDEFINED>';
    matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.fixedpatch.fixedrows = [1 Inf]; %'<UNDEFINED>';
else
    matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.npatches = 512;    
    matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.niter = 1;
end
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
matlabbatch{batch_idx}.spm.meeg.source.invertiter.crossval = [pctest Nfolds];

spm_jobman('run', matlabbatch);

% Load inversion - get cross validation error end F
Drecon=spm_eeg_load(data_file);
F=Drecon.inv{1}.inverse.crossF;
CVerr=Drecon.inv{1}.inverse.crosserr;
varargout{1}=F;
varargout{2}=CVerr;

if nargout>1
    M=Drecon.inv{1}.inverse.M;
    U=Drecon.inv{1}.inverse.U{1};
    MU=M*U;
    varargout{3}=MU;
end