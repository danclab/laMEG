function varargout=invert_msp(data_file, coreg_fname, mri_fname, mesh_fname, ...
    nas, lpa, rpa, priors, patch_size, n_temp_modes, foi, woi, Nfolds,...
    ideal_pctest, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

clear jobs
matlabbatch={};
batch_idx=1;

% Copy datafile
matlabbatch{batch_idx}.spm.meeg.other.copy.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.other.copy.outfile = coreg_fname;
batch_idx=batch_idx+1;
spm_jobman('run', matlabbatch);    

% Coregister to mesh
clear jobs
matlabbatch={};
batch_idx=1;

% Coregister simulated dataset to reconstruction mesh
matlabbatch{batch_idx}.spm.meeg.source.headmodel.D = {coreg_fname};
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
[data_dir,fname,ext]=fileparts(coreg_fname);
spatialmodesname=fullfile(data_dir, sprintf('%s_testmodes.mat',fname));
[spatialmodesname,Nmodes,pctest]=spm_eeg_inv_prep_modes_xval(coreg_fname, [], spatialmodesname, Nfolds, ideal_pctest);

% so use all vertices that will be simulated on (plus a few more) as MSP priors
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
matlabbatch{batch_idx}.spm.meeg.source.invertiter.D = {coreg_fname};
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
matlabbatch{batch_idx}.spm.meeg.source.invertiter.isstandard.custom.outinv = '';
matlabbatch{batch_idx}.spm.meeg.source.invertiter.modality = {'All'};
matlabbatch{batch_idx}.spm.meeg.source.invertiter.crossval = [pctest Nfolds];                                
batch_idx=batch_idx+1;

[a,b]=spm_jobman('run', matlabbatch);

% Load inversion - get cross validation error end F
Drecon=spm_eeg_load(coreg_fname); 
F=Drecon.inv{1}.inverse.crossF;
CVerr=Drecon.inv{1}.inverse.crosserr./Drecon.inv{1}.inverse.allrms;
varargout{1}=F;
varargout{2}=CVerr;

if nargout>1
    M=Drecon.inv{1}.inverse.M;
    U=Drecon.inv{1}.inverse.U{1};
    MU=M*U;
    varargout{3}=MU;
end