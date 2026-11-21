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
   :members: from_spm, layer, orientation_defined, validate_mesh
