function convert_ctf_fif_to_spm(orig_res4_file, mne_file, output_path, prefix, epoched)
% CONVERT_MNE_TO_SPM  Convert MNE file (fif format) to SPM
%    convert_mne_to_spm(orig_res4_file, mne_file, epoched)
%        orig_res4_file = file path of the original data 
%        mne_file = file path of the MNE (fif file) to convert
%        epoched = 0 if continuous data, 1 if epoched


% Create file path for result
[filepath,name,ext] = fileparts(mne_file);
spm_filename=fullfile(output_path, sprintf('%s%s.mat', prefix, name));
% spm_filename=output_name

% Setup SPM batch
clear jobs
matlabbatch={};
        
% Use SPM conversion
matlabbatch{1}.spm.meeg.convert.dataset = {mne_file};
if epoched
    matlabbatch{1}.spm.meeg.convert.mode.epoched.usetrials = 1;
else
    matlabbatch{1}.spm.meeg.convert.mode.continuous.readall = 1;
end
matlabbatch{1}.spm.meeg.convert.channels{1}.all = 'all';
matlabbatch{1}.spm.meeg.convert.outfile = spm_filename;
matlabbatch{1}.spm.meeg.convert.eventpadding = 0;
if epoched
    matlabbatch{1}.spm.meeg.convert.blocksize = 3276800;
else
    matlabbatch{1}.spm.meeg.convert.blocksize = 113276800;
end
matlabbatch{1}.spm.meeg.convert.checkboundary = 1;
matlabbatch{1}.spm.meeg.convert.saveorigheader = 0;
matlabbatch{1}.spm.meeg.convert.inputformat = 'autodetect';

spm_jobman('run',matlabbatch);

%%
% The fidicial locations, channel locations, and channel types cannot be
% read from the FIF file by fieldtrip, so we have to change some things
% manually
%%
load(spm_filename);

% Convert from T to fT
D.data(:,:,:)=D.data(:,:,:).*10^15;

% Get the sensor locations from the original header file
%elec = ft_read_sens(orig_res4_file);
hdr = ft_read_header(orig_res4_file);
D.sensors=[];
D.sensors.meg = hdr.grad;

% MNE adds some extra text to the channel labels that we have to remove
for i=1:length(D.channels)
    label=D.channels(i).label;
    label_parts=strsplit(label,'-');
    D.channels(i).label=label_parts{1};
    % We also need to set the units
    D.channels(i).units='fT';
end

% Get the fiducial locations from the original header file
D.fiducials = ft_convert_units(ft_read_headshape(orig_res4_file), 'mm');

% Set the trial onset time
if epoched
    hdr = ft_read_header(mne_file);
    D.timeOnset=-hdr.nSamplesPre/hdr.Fs;
end

% Save the SPM struct
save(spm_filename,'D');

% The next part has to be done with the SPM object
D1=spm_eeg_load(spm_filename);

% Set the channel types based on the original header file
hdr = ft_read_header(orig_res4_file);
origchantypes = ft_chantype(hdr);
[sel1, sel2] = spm_match_str(D1.chanlabels, hdr.label);
origchantypes = origchantypes(sel2);
if length(strmatch('unknown', origchantypes, 'exact')) ~= numel(origchantypes)
    D1.origchantypes = struct([]);
    D1.origchantypes(1).label = hdr.label(sel2);
    D1.origchantypes(1).type = origchantypes;
end
S1 = [];
S1.task = 'defaulttype';
S1.D = D1;
S1.updatehistory = 0;
D1 = spm_eeg_prep(S1);
save(D1);

% Project 3D coil locations to 2D for topography plots
D1=spm_eeg_load(spm_filename);
S1 = [];
S1.task = 'project3D';
S1.modality = 'MEG';
S1.updatehistory = 0;
S1.D = D1;
D1 = spm_eeg_prep(S1);

save(D1);

% load(spm_filename);
% D.sensors=[];
% D.sensors.meg = hdr.grad;
% save(spm_filename,'D');