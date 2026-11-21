Comparing layers with a precomputed MEG forward
===============================================

Pass a ``ForwardOperator`` to ``model_comparison`` to run EBB or MSP with signed,
fixed-orientation lead fields computed by another solver. The operator holds
one column per vertex, grouped by layer, and one row per named MEG channel.
The results follow the layer order of the supplied ``LayerSurfaceSet``.

Native SPM round trip
---------------------

This example takes a donor SPM dataset with a previously computed multilayer gain
and an independent working copy of the same recording. Both datasets must use
the same head registration, sensor response and good MEG channels. Keep the
donor's gain cache beside its header. As with native comparison, use a writable
working copy because SPM updates its inversion slot for each candidate layer.
Use a separate dataset directory for each simultaneous run. Keep the generated
``lameg_gain_*.mat`` files with the dataset for subsequent use in SPM.

.. code-block:: python

    from lameg.forward import ForwardOperator
    from lameg.laminar import model_comparison

    def compare_saved_forward(donor, working_copy, surfaces, fids, vertex, spm):
        operator = ForwardOperator.from_spm(
            donor, surfaces.get_mesh_path(), surfaces.get_layer_names()
        )
        return model_comparison(
            fids, working_copy, surfaces, method='MSP', spm_instance=spm,
            forward_operator=operator, viz=False,
            invert_kwargs=dict(priors=[vertex], patch_size=5, woi=[-25, 25]),
        )

Archived gains may contain nonzero columns at vertices with a missing dipole
direction. For a numerical round trip, ``allow_undefined_gain=True`` explicitly
preserves those columns. Use ``operator.layer(name).orientation_defined`` when
selecting sites for oriented-source comparisons, including the vertices in a
patch.

MNE-Python BEM forwards
-----------------------

Compute one free-orientation MNE forward on each laMEG layer using the same MEG
recording and head registration. The importer projects the three gain columns at
each source onto the signed dipole direction stored in the GIfTI. This example
compares layers for one CTF recording. The SPM dataset uses CTF head
coordinates and measured pickup-coil positions. The MNE forwards and the
acquisition ``Info`` come from that same recording.

.. code-block:: python

    from pathlib import Path

    from mne.transforms import invert_transform
    from lameg.forward import ForwardOperator
    from lameg.laminar import model_comparison

    def compare_ctf_bem(forward_dir, surfaces, vertices, channel_units,
                        mri_to_head_mm, info, working_copy, fids, vertex, spm):
        forwards = {
            name: Path(forward_dir) / f'{name}-bem-fwd.fif'
            for name in surfaces.get_layer_names()
        }
        mne_head_to_ctf_mm = invert_transform(info['ctf_head_t'])['trans'].copy()
        mne_head_to_ctf_mm[:3, 3] *= 1000
        operator = ForwardOperator.from_mne(
            forwards, surfaces.get_mesh_path(), vertices=vertices,
            channel_units=channel_units, mri_to_head_mm=mri_to_head_mm,
            mne_head_to_head_mm=mne_head_to_ctf_mm, info=info, ignore_ref=False,
            allow_nominal_ctf_coils=True,
        )
        return model_comparison(
            fids, working_copy, surfaces, method='MSP', spm_instance=spm,
            forward_operator=operator, viz=False,
            invert_kwargs=dict(priors=[vertex], patch_size=5, woi=[-25, 25]),
        )

Keep the source correspondence when creating the MNE source spaces. Each row
of ``vertices`` gives the zero-based MNE source-space index and vertex ID for
one vertex in *single-layer GIfTI order*. For example, ``(1, 42)`` identifies
vertex 42 in the second source space. Use a vertex ID from that space's
``vertno`` array, even when decimation makes the IDs nonconsecutive. Supply a
single array if the mapping is identical across layers, or a dictionary keyed
by layer name if it differs. Each layer's mapped positions must agree with
its GIfTI coordinates.

Both transforms take and return points in millimetres. ``mri_to_head_mm``
maps the native GIfTI coordinates into the recipient SPM head frame. An
existing native operator for this registration provides this affine as
``native.mri_to_head_mm``. ``mne_head_to_head_mm`` maps MNE head coordinates
into that same frame. The CTF example inverts the acquisition's CTF-head to
MNE-head transform and converts its translation to millimetres. For a
different recipient head frame, supply the transform for that registration.

Pass the recording's good MEG channel names and physical units in an ordered
mapping. With the same native operator, use
``dict(zip(native.channel_names, native.channel_units))``.
The importer selects the named gain rows and scales each to its recorded unit
per nA m. For CTF data with a nonzero compensation grade, set
``ignore_ref=False`` in ``mne.make_forward_solution`` and pass the same setting
and the original acquisition ``Info`` to the importer. A forward FIF alone cannot
establish whether MNE applied reference compensation. Before inversion, laMEG
checks the imported coil positions, directions and reference weights against
the recipient SPM registration.

By default laMEG requires strict sensor agreement. The example opts in to
matching MNE's nominal CTF pickup coils to the measured coils from the same
acquisition. Each nominal position must match exactly one group of measured
pickup centres within 0.15 mm Euclidean distance. At each group, laMEG checks
the combined signed response of each channel, including reference weights,
with absolute tolerance 1e-9 and relative tolerance 1e-5.

External solvers
----------------

Construct ``ForwardOperator`` directly with a physical gain in each channel's
recorded unit per nA m. Preserve signed dipole and sensor orientations. Supply
the sensor coils in head metres and the channel-by-coil transfer matrix,
including reference compensation. Supply both the original MRI surface
coordinates (mm) and the effective head coordinates (mm) used by the solver.
This allows a solver to shift a source while preserving its anatomical vertex
index and patch membership. Patch smoothing uses the requested layer surface.
The comparison checks layer names, vertex order, connectivity and orientations
before processing the first layer. It checks the recipient's sensor response
and reads the installed gain back through SPM before inversion.

.. autoclass:: lameg.forward.ForwardOperator
   :members: from_spm, from_mne, layer, orientation_defined, validate_mesh
