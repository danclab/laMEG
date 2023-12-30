function [f_vals,wois]=invert_sliding_window(prior, data_file,...
    mri_fname, mesh_fname, nas, lpa, rpa, patch_size, n_temp_modes, win_size,...
    win_overlap, foi, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');
spm_jobman('initcfg');

% Data file to load
D=spm_eeg_load(data_file);

% Create wois
times=D.time;
wois=[];
if win_overlap
    for t_idx=1:length(times)
        win_l=max(1,ceil(t_idx-win_size/2));
        win_r=min(length(times),floor(t_idx+win_size/2));
        woi=[times(win_l) times(win_r)].*1000;
        wois(t_idx,:)=woi;
    end
else
    ts=linspace(times(1),times(end),(times(end)-times(1))./(win_size/1000)).*1000;
    wois=[];
    for i=2:length(ts)
        wois(end+1,:)=[ts(i-1) ts(i)];
    end
end

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
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.D = {data_file};
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.val = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.whatconditions.all = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.invfunc = 'Classic';
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.invtype = 'MSP'; %;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.wois = wois;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.foi = foi;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.hanning = 1;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.isfixedpatch.fixedpatch.fixedfile = {patchfilename}; % '<UNDEFINED>';
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.isfixedpatch.fixedpatch.fixedrows = [1 Inf]; %'<UNDEFINED>';
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.patchfwhm = -patch_size; %% NB A fiddle here- need to properly quantify
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.mselect = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.nsmodes = Nmodes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.umodes = {spatialmodesname};
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.ntmodes = n_temp_modes;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.priors.priorsmask = {''};
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.priors.space = 0;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.restrict.radius = 32;
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.isstandard.custom.outinv = '';
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.modality = {'All'};
matlabbatch{batch_idx}.spm.meeg.source.invertiter_slidingwindow.crossval = [pctest 1];                                
batch_idx=batch_idx+1;

[a,b]=spm_jobman('run', matlabbatch);
% Get F-values for inversion
Drecon=spm_eeg_load(a{1}.D{1});                
f_vals=Drecon.inv{1}.inverse.crossF;                  
