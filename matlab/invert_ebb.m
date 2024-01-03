function varargout=invert_ebb(data_file, mri_fname, mesh_fname, ...
    nas, lpa, rpa, patch_size, n_temp_modes, foi, woi, Nfolds,...
    ideal_pctest, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

% Coregister to mesh
clear jobs
matlabbatch={};
batch_idx=1;

% Coregister simulated dataset to reconstruction mesh
matlabbatch{batch_idx}.spm.meeg.source.headmodel.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.comment = '';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.mri = {[mri_fname ',1']};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.cortex = {mesh_fname};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.iskull = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.oskull = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshes.custom.scalp = {''};
matlabbatch{batch_idx}.spm.meeg.source.headmodel.meshing.meshres = 2;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).fidname = 'nas';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(1).specification.type = nas;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).fidname = 'lpa';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(2).specification.type = lpa;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).fidname = 'rpa';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.fiducial(3).specification.type = rpa;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.coregistration.coregspecify.useheadshape = 0;
matlabbatch{batch_idx}.spm.meeg.source.headmodel.forward.eeg = 'EEG BEM';
matlabbatch{batch_idx}.spm.meeg.source.headmodel.forward.meg = 'Single Shell';
spm_jobman('run', matlabbatch);

% Setup spatial modes for cross validation
[data_dir,fname,ext]=fileparts(data_file);
spatialmodesname=fullfile(data_dir, sprintf('%s_testmodes.mat',fname));
[spatialmodesname,Nmodes,pctest]=spm_eeg_inv_prep_modes_xval(data_file, [], spatialmodesname, Nfolds, ideal_pctest);

clear jobs
matlabbatch={};
batch_idx=1;

% Source reconstruction
matlabbatch{batch_idx}.spm.meeg.source.invertiter.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.whatconditions.all = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invfunc = 'Classic';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.invtype = 'EBB'; %;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.woi = woi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.foi = foi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.hanning = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.npatches = 512;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.isfixedpatch.randpatch.niter = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.patchfwhm = -patch_size; %% NB A fiddle here- need to properly quantify
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.mselect = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.nsmodes = Nmodes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.umodes = {spatialmodesname};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.ntmodes = n_temp_modes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.priorsmask = {''};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.priors.space = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.restrict.radius = 32;
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.outinv = '';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.modality = {'All'};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.crossval = [pctest Nfolds];                                
batch_idx=batch_idx+1;

[a,b]=spm_jobman('run', matlabbatch);

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