function compute_lgainmat(data_file, out_fname, spm_path)

addpath(spm_path);

% Start SPM
spm('defaults','eeg');

D=spm_eeg_load(data_file);

val = D.val;

forward = D.inv{val}.forward;

for ind = 1:numel(forward)
    modality = forward(ind).modality;
    
    % channels
    %----------------------------------------------------------------------
    if isequal(modality, 'MEG')
        chanind = D.indchantype({'MEG', 'MEGPLANAR'}, 'GOOD');
    else
        chanind = D.indchantype(modality, 'GOOD');
    end
    
    if ~isempty(chanind)
        forward(ind).channels = D.chanlabels(chanind);
    else
        error(['No good ' modality ' channels were found.']);
    end
end

if nargin < 3
    channels = [forward(:).channels];
end

G     = {};
label = {};
for ind = 1:numel(forward)
    % create a new lead-field matrix
    %------------------------------------------------------------------

    % Head Geometry (create tesselation file)
    %------------------------------------------------------------------
    vert = forward(ind).mesh.vert;
    face = forward(ind).mesh.face;

    % normals
    %------------------------------------------------------------------
    ctx=gifti(D.inv{val}.mesh.tess_ctx);
    if isfield(ctx,'normals')
        % Transform normal vectors into right space
        M=D.inv{val}.forward(ind).fromMNI*D.inv{val}.mesh.Affine;
        norm=[ctx.normals ones(size(ctx.normals,1),1)]*inv(M')';
        norm=norm(:,1:3);
        normN = sqrt(sum(norm.^2,2));
        bad_idx=find(normN < eps);
        normN(bad_idx)=1;
        norm = bsxfun(@rdivide,norm,normN);
        norm=double(norm);
    else
        norm = spm_mesh_normals(struct('faces',face,'vertices',vert),true);
    end                      

    vol  = forward(ind).vol;

    if ischar(vol)
        vol = ft_read_vol(vol);
    end

    modality = forward(ind).modality;

    if isfield(forward, 'siunits') && forward(ind).siunits
        units = D.units(D.indchannel(forward(ind).channels));
        sens  = forward(ind).sensors;
        siunits = isempty(strmatch('unknown', units));
    else
        siunits = false;
        sens = D.inv{val}.datareg(ind).sensors;
    end

    % Forward computation
    %------------------------------------------------------------------
    [vol, sens] = ft_prepare_vol_sens(vol, sens, 'channel', forward(ind).channels);
    nvert = size(vert, 1);

    spm('Pointer', 'Watch');drawnow;
    spm_progress_bar('Init', nvert, ['Computing ' modality ' leadfields']); drawnow;
    if nvert > 100, Ibar = floor(linspace(1, nvert,100));
    else Ibar = [1:nvert]; end

    PARALLEL=1; %% USE PARFOR
    if ~isequal(ft_voltype(vol), 'interpolate')
        if PARALLEL==0,
            Gxyz = zeros(length(forward(ind).channels), 3*nvert);
            for i = 1:nvert

                if siunits
                    Gxyz(:, (3*i- 2):(3*i))  = ft_compute_leadfield(vert(i, :), sens, vol,...
                        'dipoleunit', 'nA*m', 'chanunit', units);
                else
                    Gxyz(:, (3*i- 2):(3*i))  = ft_compute_leadfield(vert(i, :), sens, vol);
                end

                if ismember(i, Ibar)
                    spm_progress_bar('Set', i); drawnow;
                end

            end
        else %% RUN IN PARALLEL
            Gxyz = zeros(nvert, length(forward(ind).channels),3);

            parfor i = 1:nvert


                if siunits
                    Gxyz(i,:,:)  = ft_compute_leadfield(vert(i, :), sens, vol,...
                        'dipoleunit', 'nA*m', 'chanunit', units);
                else
                    Gxyz(i,:,:)  = ft_compute_leadfield(vert(i, :), sens, vol);
                end
            end

            Gxyz=shiftdim(Gxyz,1);
            Gxyz=reshape(Gxyz,length(forward(ind).channels),3*nvert);

        end; %% if PARALLEL
    else
        if siunits
            Gxyz = ft_compute_leadfield(vert, sens, vol, 'dipoleunit', 'nA*m', 'chanunit', units);
        else
            Gxyz = ft_compute_leadfield(vert, sens, vol);
        end
    end

    spm_progress_bar('Clear');
    spm_progress_bar('Init', nvert, ['Orienting ' modality ' leadfields']); drawnow;

    G{ind} = zeros(size(Gxyz, 1), size(Gxyz, 2)/3);
    for i = 1:nvert

        G{ind}(:, i) = Gxyz(:, (3*i- 2):(3*i))*norm(i, :)';

        if ismember(i, Ibar)
            spm_progress_bar('Set', i); drawnow;
        end

    end

    % condition the scaling of the lead-field
    %--------------------------------------------------------------------------
    [Gs, scale] = spm_cond_units(G{ind});

    if siunits && abs(log10(scale))>2
        warning(['Scaling expected to be 1 for SI units, actual scaling ' num2str(scale)]);
        G{ind} = Gs;
    else
        scale = 1;
    end

    spm_progress_bar('Clear');

    spm('Pointer', 'Arrow');drawnow;

    label = [label; forward(ind).channels(:)];

    forward(ind).scale = scale;
end

if numel(G)>1
    G = cat(1, G{:});
else
    G = G{1};
end

% Save
%----------------------------------------------------------------------
save(out_fname, 'G', 'label', '-v7.3');
