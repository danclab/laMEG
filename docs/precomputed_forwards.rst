Using a precomputed MEG forward
===============================

Pass the same ``forward_operator`` to a direct inversion, a sliding-window
inversion, or a layer comparison. It holds one gain row per named MEG channel
and one signed column per vertex, grouped by layer. Each analysis writes to
its SPM inversion slot. For comparisons between solvers, give each run its
own writable copy of the same recording in a separate directory.

MNE BEM and native single-shell on the same vertices
----------------------------------------------------

If you have an MNE BEM solution, use ``compute_mne_forward`` to solve on the
original laMEG surfaces. After coregistering the SPM reference to the multilayer
mesh, pass the BEM solution, acquisition Info and MNE head/MRI transform as in
Tutorial 10. If you already have a forward for each layer, use
``prepare_mne_forward`` with a dictionary of layer names to MNE Forward objects
or FIF paths, as in the example below. Both functions check the registered
source positions and prepare a surface set containing the anatomical locations
retained at every depth. For example, if MNE excludes a pial vertex outside
the inner skull, its matching white vertex is also omitted from the prepared
surfaces. Use those surfaces for both solvers. For coincident positions that
make matching ambiguous when importing existing forwards, supply the source-space
and vertex IDs via ``vertices``.

The example compares seeded MSP across layers and time windows for a CTF
recording. ``reference`` is an SPM dataset already coregistered to the
original multilayer mesh. ``info`` is the original MNE acquisition Info,
including the reference channels used to compute the compensated forwards.

.. code-block:: python

    from lameg.forward import prepare_mne_forward
    from lameg.laminar import sliding_window_model_comparison

    def compare_solver_windows(forwards, surfaces, reference, info, output_dir,
                               bem_copy, native_copy, fids, vertex, spm):
        # Take the intersection across layers once. Both solvers will then use the same
        # cortical locations, even if the BEM excluded a vertex from only one layer.
        prepared = prepare_mne_forward(
            forwards, surfaces, output_dir, reference=reference, info=info,
            ignore_ref=False, allow_nominal_ctf_coils=True,
        )
        options = dict(
            spm_instance=spm, viz=False,
            invert_kwargs=dict(wois=[[-50, -25], [-25, 0], [0, 25]], patch_size=5),
        )
        # vertex is the original within-layer ID. Each call maps it to the kept
        # vertex before running MSP in the specified windows.
        bem = sliding_window_model_comparison(
            vertex, fids, bem_copy, prepared.surfaces,
            forward_operator=prepared.operator, **options,
        )
        # Use SPM's native gain on the same prepared surfaces and an independent copy
        # of the recording so that this run preserves the BEM inversion on bem_copy.
        single_shell = sliding_window_model_comparison(
            vertex, fids, native_copy, prepared.surfaces, **options,
        )
        return bem, single_shell, prepared.manifest

Each result is a pair containing the layer-by-window free energies and the
window bounds in milliseconds. Rows follow ``surfaces.get_layer_names()``.
Supply the seed using its zero-based
within-layer vertex ID on the **original** mesh. If the seed is excluded,
laMEG raises an error. It maps a retained seed to the corresponding row on
the prepared mesh automatically. Direct inversions on the combined mesh
accept MSP priors in original layer-major order. Use
``prepared.support.original_vertices`` to map compact source rows within
each layer back to anatomical vertex IDs. The JSON file at
``prepared.manifest`` records the vertex correspondence and exclusions.

For this example the SPM recipient uses CTF head coordinates. Preparation
reads the MRI registration and channel units from the SPM header, and obtains
the MNE-to-CTF head transform from the acquisition Info. For other coordinate
frames, supply the measured transform explicitly. The example enables nominal
CTF coil matching because MNE uses nominal pickup positions and this SPM file
has measured positions for the same recording. Strict sensor matching is the
default. With this option, each nominal pickup must match a unique group of
measured coils within 0.15 mm. Each channel's signed response, including its
reference weights, is compared at each group with absolute tolerance 1e-9 and
relative tolerance 1e-5. For compensated CTF acquisitions, ``ignore_ref=False``
must match the setting used to compute the forward.

Direct inversions
-----------------

Direct inversions require a working dataset registered to the mesh you will
invert, including any vertex exclusions. The example below registers
``prepared.surfaces`` in the working dataset before fitting EBBlayer. The
default ``layer_name=None`` selects the combined multilayer mesh. For an
individual layer, pass the same layer name and inversion slot to both
registration and inversion. The layer-comparison functions perform this step
internally for each candidate layer.

.. code-block:: python

    from lameg.invert import coregister, invert_ebb_layer

    def fit_multilayer_ebb(working_copy, prepared, fids, spm):
        # Register the combined mesh to this recording before fitting across layers.
        coregister(fids, working_copy, prepared.surfaces, spm_instance=spm, viz=False)
        # Rows of the returned reconstruction matrix follow the kept vertices of
        # each layer in order. prepared.manifest records their original IDs.
        return invert_ebb_layer(
            working_copy, prepared.surfaces, forward_operator=prepared.operator,
            woi=[-25, 25], spm_instance=spm, return_mu_matrix=True, viz=False,
        )

The call returns free energy, cross-validation error, and the source
reconstruction matrix. To identify an original cortical location from row
``r`` of the combined matrix, set ``ids = prepared.support.original_vertices``
and compute ``layer_index, i = divmod(r, len(ids))``. The anatomical vertex
is ``ids[i]`` on ``prepared.surfaces.get_layer_names()[layer_index]``. Keep
``source_support.json`` with results exported for later analysis.

The same ``forward_operator`` keyword works with ``invert_ebb``, ``invert_ebb_layer``, ``invert_msp``,
and their three ``invert_sliding_window_*`` counterparts. A supplied operator
is installed before spatial-mode preparation and is used for all time windows
in that inversion. SPM keeps the attached gain in that slot. Keep its
``lameg_gain_*.mat`` cache beside the dataset.

Native SPM round trip
---------------------

This example takes a donor SPM dataset with a previously computed multilayer gain
and an independent working copy of the same recording. Both must use the same
head registration, sensor response and good MEG channels. Keep the donor's
gain cache beside its header.

.. code-block:: python

    from lameg.forward import ForwardOperator
    from lameg.laminar import model_comparison

    def compare_saved_forward(donor, working_copy, surfaces, fids, vertex, spm):
        # Read the gain referenced by the donor, including its channel order and
        # dipole signs, before selecting the matching layer for each inversion.
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

Importing MNE forwards with known vertex correspondence
-------------------------------------------------------

Use ``ForwardOperator.from_mne`` directly when all requested vertices are
present and the correspondence is already known. Each row of ``vertices``
gives the MNE source-space index and vertex ID for one vertex in single-layer
GIfTI order. For example, ``(1, 42)`` identifies vertex 42 in the second source
space. Supply a single array if the mapping is identical across layers, or a
dictionary keyed by layer name if it differs. Pass the recording's good MEG
channel names and physical units as an ordered mapping. The importer projects
free-orientation gains onto the signed dipole directions in the GIfTI and
scales each channel to its recorded unit per nA m.

Both coordinate transforms take and return points in millimetres.
``mri_to_head_mm`` maps native GIfTI MRI coordinates into the recipient SPM
head frame. ``mne_head_to_head_mm`` maps MNE head coordinates into the same
frame. A coregistered native operator provides ``native.mri_to_head_mm`` and
``dict(zip(native.channel_names, native.channel_units))``.

DUNEuro FEM on the same cortical layers
---------------------------------------

Tutorial 10 uses the subject's CHARM mesh to simulate each cortical depth
with FEM. Install the Gmsh reader with ``pip install -e '.[fem]'`` from the
laMEG checkout. Build the
`DUNEuro Python bindings <https://www.medizin.uni-muenster.de/duneuro/startseite.html>`_
with tetrahedral continuous Galerkin support for the Python interpreter running
the notebook. Verify that ``import duneuropy`` succeeds in that environment.
For an MPI-enabled DUNE build, initialize its matching MPI runtime first
(e.g., ``from mpi4py import MPI``). Serial builds can run directly.
Whole-head calculations need memory for the volume solver and its transfer
matrix, which stores one double per active pickup and FEM degree of freedom.
SPM performs the simulations and inversions.

Pass the coregistered SPM recording as ``reference``. This supplies the
MRI-to-head transform, physical coil geometry and channel weights. The
``LayerSurfaceSet`` supplies the positions and signed dipole directions.
The solver uses the CHARM tetrahedra and the
`SimNIBS isotropic tissue conductivities <https://simnibs.github.io/simnibs/build/html/documentation/conductivity.html>`_.
Supply a tag-to-conductivity mapping in S/m to override them. If CHARM's T1
and the laMEG GIfTIs use different scanner coordinates, supply their measured
rigid registration in millimetres as ``mri_to_fem_mm``.

.. code-block:: python

    from lameg.forward import compute_duneuro_forward
    from lameg.invert import coregister

    # Run coregister once for this recording if it has not yet been registered.
    coregister(fid_coords, base_fname, surf_set, iskull_fname=skull_fname, spm_instance=spm)
    prepared = compute_duneuro_forward(
        surf_set, charm_fname, reference=base_fname, output_dir=tmp_dir
    )
    fem = prepared.operator
    surfaces = prepared.surfaces

The fitted patch-based St. Venant model uses the signed cortical link vectors.
DUNEuro currently marks this source model experimental.
A column must lie in CHARM gray matter at every requested depth and belong to
a surviving cortical triangle. This can remove substantial portions of the
cortex at the pial or white boundary. Inspect the count and the original
vertex IDs in ``prepared.manifest`` when interpreting a comparison. The
simulator and all three inversions in Tutorial 10 use this final surface set.

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
