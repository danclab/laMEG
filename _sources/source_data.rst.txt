Source data and HDF5 schema
===========================

This page documents the analysis-ready source-data interface used by laMEG.
It describes the public API, the HDF5 file format, and the distinction between
reconstructed cortical-depth layers and BigBrain-informed laminar estimates.

Overview
--------

Sliding-window source reconstructions can be exported to a self-describing
HDF5 file with
:func:`lameg.source.export_source_time_series_hdf5`.

The resulting file can be accessed with
:class:`lameg.source.LaminarSourceData` without loading the complete source
array into memory.

The main source dataset has one of two axis orders::

    layer, column, time

or::

    layer, column, time, trial

Here, ``layer`` denotes one surface in the reconstructed multilayer cortical
geometry and ``column`` denotes corresponding vertices across those surfaces.

Exporting source data
---------------------

The exporter derives the number of reconstructed layers, number of cortical
columns, and normalized cortical-depth coordinates from a
:class:`lameg.surf.LayerSurfaceSet`.

For example::

    from lameg.source import export_source_time_series_hdf5
    from lameg.surf import LayerSurfaceSet

    surf_set = LayerSurfaceSet("sub-001", 11)

    source_fname = export_source_time_series_hdf5(
        data_fname="spm_data.mat",
        out_fname="sub-001_source.h5",
        surf_set=surf_set,
    )

The exporter verifies that the source geometry in the inverse solution is
consistent with the supplied ``LayerSurfaceSet``.

The cortical-depth convention used throughout the source format is::

    0 = pial boundary
    1 = grey/white boundary

Consequently, ``layer_depth`` increases from the pial surface towards white
matter.

Reading source data
-------------------

``LaminarSourceData`` should normally be used as a context manager so that the
underlying HDF5 file is closed automatically::

    from lameg.source import LaminarSourceData

    with LaminarSourceData("sub-001_source.h5") as source:
        print(source.shape)
        print(source.axis_order)
        print(source.layer_depth)

        # All reconstructed layers for one cortical column.
        column_data = source.layer(column=6620)

        # One reconstructed layer over a selected time interval.
        layer_data = source.layer(
            layer=4,
            time=(-75, -25),
        )

Selections are read directly from the HDF5 file. Integer selectors remove
their corresponding axis, whereas slices and ``None`` preserve it.

For files containing an explicit trial axis, a trial can be selected with the
``trial`` argument.

Layers and laminae
------------------

``layer`` and ``lamina`` have deliberately different meanings in the laMEG
source API.

``layer``
    A surface in the reconstructed multilayer cortical geometry. Layers are
    depth samples on which the electromagnetic inverse solution is defined.

``lamina``
    A BigBrain-informed histological compartment obtained by transforming
    reconstructed cortical-depth activity using column-specific laminar
    weights.

A BigBrain mapping can be added to an existing source file without repeating
the source reconstruction::

    from lameg.source import add_bigbrain_mapping
    from lameg.surf import LayerSurfaceSet

    surf_set = LayerSurfaceSet("sub-001", 11)

    add_bigbrain_mapping(
        "sub-001_source.h5",
        surf_set,
    )

Mapped activity can then be accessed on demand::

    from lameg.source import LaminarSourceData

    with LaminarSourceData("sub-001_source.h5") as source:
        print(source.laminae)

        valid_columns = source.bigbrain_valid_columns

        lamina_v = source.lamina(
            lamina="V",
            column=6620,
        )

.. warning::

   ``LaminarSourceData.lamina()`` does not represent a direct measurement of
   cytoarchitectonic cortical layers. It applies a BigBrain-informed
   transformation to reconstructed cortical-depth activity. The
   interpretability of lamina-specific estimates therefore depends on the
   spatial information available in the MEG or OPM-MEG data, the forward
   model, cortical geometry, and the assumptions of the source reconstruction.

HDF5 schema
-----------

The current source-data schema version is ``1.0``.

The schema version is stored independently of the laMEG package version in
the root attribute::

    lameg_schema_version

``LaminarSourceData`` validates the schema when a file is opened and rejects
unsupported schema versions rather than silently interpreting them.

Required root attributes
~~~~~~~~~~~~~~~~~~~~~~~~

``lameg_schema_version``
    Version of the laMEG source-data schema.

``lameg_version``
    Version of laMEG that created the file.

``axis_order``
    Either ``layer,column,time`` or ``layer,column,time,trial``.

``source_order``
    Ordering of sources in the original inverse representation. Schema v1
    requires ``layer-major``.

``n_layers``
    Number of reconstructed cortical-depth surfaces.

``n_columns``
    Number of corresponding cortical columns.

Files produced by ``export_source_time_series_hdf5`` also contain provenance
information including ``subject_id``, ``orientation_method``,
``fixed_orientation``, ``source_geometry``, ``n_windows``, ``inversion_idx``,
``overlap_combination``, and ``export_strategy``.

Core datasets
~~~~~~~~~~~~~

``source_ts``
    Shape ``(layer, column, time)`` or
    ``(layer, column, time, trial)``. Contains the reconstructed source time
    series.

``time_ms``
    Shape ``(time,)``. Strictly increasing sample times in milliseconds.

``layer_depth``
    Shape ``(layer,)``. Normalized reconstructed cortical depths using the
    convention ``0=pial,1=white``.

``source_vertices``
    Shape ``(layer, column, 3)``, when present. Source coordinates associated
    with the SPM forward model.

``woi_ms``
    Shape ``(window, 2)``, when present. Sliding inversion windows in
    milliseconds.

``window_count``
    Shape ``(time,)``, when present. Number of inversion windows contributing
    to each reconstructed time sample.

BigBrain mapping
~~~~~~~~~~~~~~~~

When a BigBrain mapping has been added, the file contains a ``bigbrain``
group.

``bigbrain/edges``
    Shape ``(column, 7)``. Column-specific normalized boundaries defining
    laminae I--VI.

``bigbrain/weights``
    Shape ``(column, 6, layer)``. Linear transformation weights from
    reconstructed cortical-depth samples to BigBrain-informed laminar means.

``bigbrain/labels``
    The labels ``I``, ``II``, ``III``, ``IV``, ``V``, and ``VI``.

``bigbrain/valid_columns``
    Shape ``(column,)``. Boolean mask identifying cortical columns for which a
    valid BigBrain mapping is available.

Columns for which the anatomical mapping cannot be computed are explicitly
marked invalid rather than being assigned a neighbouring or population-average
mapping. Analyses using BigBrain-informed laminar estimates should therefore
inspect ``bigbrain_valid_columns``.

API reference
-------------

.. autoclass:: lameg.source.LaminarSourceData
   :members:
   :inherited-members:

.. autofunction:: lameg.source.export_source_time_series_hdf5

.. autofunction:: lameg.source.add_bigbrain_mapping

.. autofunction:: lameg.source.load_source_time_series
