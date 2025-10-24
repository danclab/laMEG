"""
Tools for generating, managing, and postprocessing laminar cortical surfaces.

This module implements a complete workflow for constructing, manipulating, and visualizing
laminar surface meshes derived from FreeSurfer reconstructions. It provides a high-level API
for end-to-end postprocessing, including coordinate transformations, hemisphere merging,
downsampling, and computation of laminar dipole orientations.

Public API
----------
Classes
--------
LayerSurfaceSet
    High-level interface for organizing, validating, and manipulating laminar surface hierarchies.

Functions
---------
interpolate_data
    Interpolate scalar or vector data between aligned surface meshes.
convert_fsaverage_to_native
    Convert vertices from the fsaverage template space into the subject's native surface space.
convert_native_to_fsaverage
    Convert vertices from a subject's native space to the fsaverage template.

Internal utilities handle surface file I/O, mesh decimation, transformation tracking, and
FreeSurfer compatibility.
"""

# pylint: disable=C0302
import copy
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree  # pylint: disable=E0611
from vtkmodules.util.numpy_support import vtk_to_numpy
# pylint: disable=E0611
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkFiltersCore import vtkDecimatePro

from lameg.util import big_brain_proportional_layer_boundaries, check_freesurfer_setup


# pylint: disable=R0902
class LayerSurfaceSet:
    """
    Object-oriented interface for managing subject-specific laminar cortical surfaces.

    This class provides standardized access to the laminar surface hierarchy generated from
    FreeSurfer reconstructions, stored under:
        $SUBJECTS_DIR/<subj_id>/surf/laminar

    It handles file naming, metadata sidecars, validation, and mappings between processing stages
    (e.g., 'converted' -> 'nodeep' -> 'combined' -> 'ds'). The class also provides convenience
    functions for loading, saving, and verifying consistency across layers and hemispheres.

    Parameters
    ----------
    subj_id : str
        Subject identifier corresponding to the FreeSurfer reconstruction directory.
    n_layers : int
        Number of cortical layers (including white and pial surfaces).
    subjects_dir : str, optional
        Path to the FreeSurfer SUBJECTS_DIR. If None, it is read from the environment variable.

    Attributes
    ----------
    subj_id : str
        The subject identifier.
    n_layers : int
        Number of layers managed by this surface set.
    layer_spacing : np.ndarray
        Proportional thickness values between 0 (white matter) and 1 (pial).
    subjects_dir : str
        Path to the FreeSurfer subjects directory.
    subj_dir : str
        Path to the subject's directory within SUBJECTS_DIR.
    mri_file : str
        Path to the subject's mri/orig.nii, converted from mri/orig.mgz
    surf_dir : str
        Path to the subject's FreeSurfer surface directory.
    laminar_surf_dir : str
        Path to the laminar surface output directory.

    Notes
    -----
    - File naming follows the convention:
        <hemi>.<layer>.<stage>[.<orientation>[.<fixed>]].gii
      where each stage represents a processing step (e.g., 'raw', 'converted', 'nodeep',
      'combined', 'ds').
    - JSON sidecars store transformation matrices, removed vertices, and other metadata.
    - The class provides validation and mapping functions to ensure geometric and topological
      consistency across all processing stages.
    """

    def __init__(self, subj_id, n_layers, subjects_dir=None):
        """
        Initialize a `LayerSurfaceSet` instance for a given subject.

        This constructor sets up paths and metadata for managing laminar surface files
        derived from a FreeSurfer reconstruction. It verifies the existence of the subject
        directory and defines the expected hierarchy under `$SUBJECTS_DIR/<subj_id>/surf/laminar`.

        Parameters
        ----------
        subj_id : str
            Subject identifier corresponding to the FreeSurfer reconstruction directory.
        n_layers : int
            Number of cortical layers (including pial and white matter surfaces).
        subjects_dir : str, optional
            Path to the FreeSurfer `SUBJECTS_DIR`. If not provided, the environment variable
            `SUBJECTS_DIR` is used.

        Raises
        ------
        EnvironmentError
            If neither `subjects_dir` is provided nor the environment variable `SUBJECTS_DIR`
            is set.
        FileNotFoundError
            If the subject directory does not exist or if FreeSurfer reconstruction outputs
            are missing.

        Notes
        -----
        - The laminar surface directory is assumed to be located at:
              $SUBJECTS_DIR/<subj_id>/surf/laminar
        - The cortical layer spacing is defined linearly from white matter (0.0) to pial (1.0).
        """
        self.subj_id = subj_id
        self.n_layers = n_layers
        self.layer_spacing = np.linspace(1, 0, self.n_layers)
        self.subjects_dir = subjects_dir or os.getenv('SUBJECTS_DIR')
        if self.subjects_dir is None:
            raise EnvironmentError("SUBJECTS_DIR is not set and no subjects_dir was provided.")

        self.subj_dir = os.path.join(self.subjects_dir, self.subj_id)
        if not os.path.exists(self.subj_dir):
            raise FileNotFoundError(
                f"Subject directory not found: {self.subj_dir}. "
                f"Please ensure that FreeSurfer's 'recon-all' has been successfully run for this "
                f"subject."
            )

        self.mri_file = os.path.join(self.subj_dir, 'mri', 'orig.nii')
        self.surf_dir = os.path.join(self.subj_dir, 'surf')
        self.laminar_surf_dir = os.path.join(self.surf_dir, 'laminar')

    def __repr__(self):
        """
        Return a concise string representation of the `LayerSurfaceSet` instance.

        The representation includes the subject identifier, number of cortical layers,
        and the path to the laminar surface directory.

        Returns
        -------
        str
            Formatted string summarizing the subject ID, number of layers, and laminar surface
            path.
        """
        return f"<LayerSurfaceSet subj_id={self.subj_id!r}, " \
               f"n_layers={self.n_layers}, dir={self.laminar_surf_dir!r}>"

    def _get_fname(self, layer_name, stage='raw', hemi=None, orientation=None, fixed=None):
        """
        Construct a standardized filename for a laminar surface mesh.

        This function assembles a full file path for a surface at a given processing
        stage, hemisphere, and orientation setting within the laminar surface directory.

        Parameters
        ----------
        layer_name : str
            Name of the cortical layer (e.g., 'pial', 'white', or intermediate layer label).
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'raw'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. Required for per-hemisphere stages ('raw', 'converted',
            'nodeep').
        orientation : str, optional
            Orientation method (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Used only for downsampled surfaces.
        fixed : bool, optional
            Whether the surface uses fixed dipole orientations. If True, 'fixed' is appended;
            if False, 'not_fixed' is appended.

        Returns
        -------
        str
            Absolute path to the constructed surface file within the laminar surface directory.
        """
        parts = []
        if hemi:
            parts.append(hemi)
        parts.append(layer_name)
        parts.append(stage)
        if orientation:
            parts.append(orientation)
            if fixed is not None:
                parts.append('fixed' if fixed else 'not_fixed')
        fname = '.'.join(parts) + '.gii'
        return os.path.join(self.laminar_surf_dir, fname)

    def get_mesh_path(self, layer_name=None, stage='ds', hemi=None, orientation='link_vector',
                      fixed=True):
        """
        Return the full path to a single-layer or multilayer surface mesh file.

        This function constructs the absolute path to a laminar surface file for a given
        layer, processing stage, hemisphere, and orientation configuration. If no layer name
        is provided, it returns the path to the combined multilayer surface.

        Parameters
        ----------
        layer_name : str or None, optional
            Name of the cortical layer (e.g., 'white', 'pial', or an intermediate layer label such
            as '0.333'). If None, the function returns the path to the combined multilayer surface
            (e.g., 'multilayer.<n_layers>').
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'ds'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. Required for hemisphere-specific stages ('raw', 'converted',
            'nodeep').
        orientation : str, optional
            Orientation method (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Used only for downsampled surfaces. Default is 'link_vector'.
        fixed : bool, optional
            Whether the surface uses fixed dipole orientations. If True, 'fixed' is appended to the
            filename. Default is True.

        Returns
        -------
        str
            Absolute path to the requested surface mesh file.
        """
        if layer_name is None:
            layer_name = f"multilayer.{self.n_layers}"
        path = self._get_fname(layer_name, stage=stage, hemi=hemi, orientation=orientation,
                               fixed=fixed)
        return path

    def load_meta(self, layer_name=None, stage='raw', hemi=None, orientation=None, fixed=None):
        """
        Load metadata associated with a laminar surface file.

        This function retrieves the JSON sidecar metadata corresponding to a surface mesh
        at a specified processing stage, hemisphere, and orientation configuration. If the
        metadata file does not exist, an empty dictionary is returned.

        Parameters
        ----------
        layer_name : str, optional
            Name of the cortical layer (e.g., 'pial', 'white', or an intermediate layer label such
            as '0.333'). If None, the function attempts to load metadata for the multilayer
            combined surface.
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'raw'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. Required for hemisphere-specific stages ('raw', 'converted',
            'nodeep'). If None, loads metadata for the combined surface.
        orientation : str, optional
            Orientation method (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Used only for downsampled surfaces.
        fixed : bool, optional
            Whether the surface uses fixed dipole orientations. If True, loads metadata from the
            corresponding 'fixed' file variant.

        Returns
        -------
        dict
            Dictionary containing the surface metadata. Returns an empty dictionary if no metadata
            file is found.
        """
        meta_path = self.get_mesh_path(
            layer_name=layer_name,
            stage=stage,
            hemi=hemi,
            orientation=orientation,
            fixed=fixed
        ).replace('.gii', '.json')

        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def update_meta(self, layer_name=None, stage='raw', hemi=None, orientation=None, fixed=None,
                    updates=None):
        """
        Update or create the metadata JSON sidecar for a surface file.

        This function loads the existing metadata for a given surface, applies updates from
        a provided dictionary, and writes the modified metadata back to disk. If the metadata
        file does not exist, it is created automatically. The modification timestamp is always
        updated.

        Parameters
        ----------
        layer_name : str, optional
            Name of the cortical layer (e.g., 'pial', 'white', or intermediate layer label such as
            '0.333'). If None, the function targets the combined multilayer surface.
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'raw'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. Required for hemisphere-specific stages ('raw', 'converted',
            'nodeep').
        orientation : str, optional
            Orientation method (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Used only for downsampled surfaces.
        fixed : bool, optional
            Whether the surface uses fixed dipole orientations. If True, updates metadata for the
            corresponding 'fixed' file variant.
        updates : dict, optional
            Dictionary of key-value pairs to add or modify in the metadata. If None, no update is
            performed.

        Notes
        -----
        - A timestamp field ('modified_at') is automatically added or updated.
        - The directory structure for the metadata file is created if it does not exist.
        """
        if updates is None:
            return

        meta = self.load_meta(layer_name, stage, hemi, orientation, fixed)
        meta.update(updates)
        meta['modified_at'] = datetime.now().isoformat()

        meta_path = self.get_mesh_path(
            layer_name=layer_name,
            stage=stage,
            hemi=hemi,
            orientation=orientation,
            fixed=fixed
        ).replace('.gii', '.json')

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as file:
            json.dump(meta, file, indent=2)

    def get_layer_names(self):
        """
        Return the list of cortical layer identifiers for this subject.

        This function generates the ordered list of surface layer names corresponding to
        the proportional spacing defined in `self.layer_spacing`. The first and last
        layers are labeled as 'pial' and 'white', respectively, while intermediate layers
        are represented by fractional depth values between 0 and 1.

        Returns
        -------
        list of str
            Ordered list of layer identifiers (e.g., ['pial', '0.900', '0.800', ..., 'white']).
        """
        layer_names = []
        for layer in self.layer_spacing:
            if layer == 1:
                layer_names.append('pial')
            elif layer == 0:
                layer_names.append('white')
            else:
                layer_names.append(f'{layer:.3f}')
        return layer_names

    def load(self, layer_name=None, stage='raw', hemi=None, orientation=None, fixed=None):
        """
        Load a GIFTI surface mesh for a specified cortical layer and processing stage.

        This function retrieves and loads a surface file corresponding to a given layer,
        hemisphere, and processing stage. If `layer_name` is None, the multilayer combined
        surface (e.g., 'multilayer.<n_layers>') is loaded. Optionally, orientation and fixation
        metadata can be specified to load downsampled or orientation-specific variants.
        Combined surfaces (both hemispheres) are loaded when `hemi` is None.

        Parameters
        ----------
        layer_name : str, optional
            Name of the cortical layer (e.g., 'pial', 'white', intermediate label such as '0.333',
            or None to load the multilayer combined surface).
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'raw'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. If None, the function loads the combined surface file.
        orientation : {'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps'}, optional
            Orientation method used for dipole estimation (applies only to downsampled surfaces).
        fixed : bool, optional
            Whether a fixed-orientation variant of the surface should be loaded.

        Returns
        -------
        nib.GiftiImage
            The loaded GIFTI surface object.

        Raises
        ------
        FileNotFoundError
            If the requested surface file does not exist.
        """
        path = self.get_mesh_path(
            layer_name=layer_name,
            stage=stage,
            hemi=hemi,
            orientation=orientation,
            fixed=fixed
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Surface not found: {path}")
        return nib.load(path)

    def save(self, gifti_obj, layer_name=None, stage='raw', hemi=None, orientation=None,
             fixed=None, meta=None):
        """
        Save a GIFTI surface mesh and its associated metadata.

        This function saves a GIFTI surface object to the laminar surface directory using the
        standardized naming convention defined by the layer, stage, hemisphere, and orientation.
        A corresponding JSON sidecar file is also written, containing metadata and a timestamp
        of the last modification. If no metadata is provided, a minimal file with a modification
        timestamp is created.

        Parameters
        ----------
        gifti_obj : nib.GiftiImage
            The GIFTI surface object to save.
        layer_name : str, optional
            Name of the cortical layer (e.g., 'pial', 'white', '0.333', or None for multilayer
            surfaces).
        stage : str, optional
            Processing stage of the surface (e.g., 'raw', 'converted', 'nodeep', 'combined', 'ds').
            Default is 'raw'.
        hemi : {'lh', 'rh'}, optional
            Hemisphere identifier. If None, saves the combined surface.
        orientation : {'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps'}, optional
            Orientation method, if applicable (used for downsampled surfaces).
        fixed : bool, optional
            Whether a fixed-orientation variant should be saved.
        meta : dict, optional
            Optional metadata dictionary to be saved alongside the surface as a JSON file.
            If None, only a timestamp entry is written.

        Notes
        -----
        - The GIFTI file is saved in the laminar surface directory for the subject.
        - Metadata is stored in a `.json` sidecar file with the same base name as the surface file.
        - The metadata file always includes a `modified_at` timestamp field.
        """
        os.makedirs(self.laminar_surf_dir, exist_ok=True)
        path = self.get_mesh_path(
            layer_name=layer_name,
            stage=stage,
            hemi=hemi,
            orientation=orientation,
            fixed=fixed
        )
        nib.save(gifti_obj, path)

        # Save optional metadata as JSON sidecar
        if meta is None:
            meta = {}

        meta_path = path.replace('.gii', '.json')
        meta['modified_at'] = datetime.now().isoformat()
        with open(meta_path, 'w', encoding='utf-8') as file:
            json.dump(meta, file, indent=2)

    # pylint: disable=R0912
    def map_between_stages(self, layer_name,
                           from_stage='ds', from_hemi=None,
                           to_stage='converted', to_hemi=None):
        """
        Map vertex indices between different surface processing stages.

        This function computes vertex correspondences between surfaces at different
        stages of the laminar processing pipeline (e.g., 'converted', 'nodeep',
        'combined', 'ds'). It handles stage-specific transformations such as vertex
        removal, hemisphere concatenation/splitting, and geometric downsampling.
        The mapping is returned as an integer index array that maps vertices in the
        `from_stage` surface to their nearest counterparts in the `to_stage` surface.

        Parameters
        ----------
        layer_name : str
            Name of the cortical layer (e.g., 'pial', 'white', or intermediate layer such as
            '0.333').
        from_stage : str, optional
            Source processing stage from which vertices are mapped (e.g., 'converted', 'nodeep',
            'combined', 'ds'). Default is 'ds'.
        from_hemi : {'lh', 'rh'}, optional
            Hemisphere for the source surface. Must be specified when mapping from per-hemisphere
            stages ('converted' or 'nodeep').
        to_stage : str, optional
            Target processing stage to which vertices are mapped (e.g., 'converted', 'nodeep',
            'combined', 'ds'). Default is 'converted'.
        to_hemi : {'lh', 'rh'}, optional
            Hemisphere for the target surface. Must be specified when mapping to per-hemisphere
            stages ('converted' or 'nodeep').

        Returns
        -------
        idx : np.ndarray
            Integer array mapping vertex indices from `from_stage` to `to_stage`.

        Raises
        ------
        ValueError
            If invalid hemisphere combinations are specified or if mapping across hemispheres is
            attempted.
        FileNotFoundError
            If one of the required surface files does not exist.

        Notes
        -----
        - Stage-specific mappings are handled as follows:
            * **'converted' -> 'nodeep'** - applies vertex removal based on metadata.
            * **'nodeep' -> 'combined'** - concatenates hemispheres and offsets right-hemisphere
              indices.
            * **'combined' -> 'nodeep'** - splits the combined mesh into hemispheres.
            * **'combined' -> 'ds'** - maps vertices geometrically via nearest-neighbour search.
        - All other transitions fall back to geometric nearest-neighbour mapping.
        """

        # --- basic sanity ---
        if from_hemi and from_stage in ('combined', 'ds'):
            raise ValueError(f"Cannot select single hemisphere from {from_stage}")
        if not from_hemi and from_stage in ('converted', 'nodeep'):
            raise ValueError(f"Specify from_hemi ('lh' or 'rh') when mapping from {from_stage}.")
        if to_hemi and to_stage in ('combined', 'ds'):
            raise ValueError(f"Cannot select single hemisphere for {to_stage}")
        if not to_hemi and to_stage in ('converted', 'nodeep'):
            raise ValueError(f"Specify to_hemi ('lh' or 'rh') when mapping to {to_stage}.")
        if from_hemi and to_hemi and from_hemi != to_hemi:
            raise ValueError("Cannot map across hemispheres")

        # --- load relevant surfaces ---
        from_surf = self.load(layer_name, stage=from_stage, hemi=from_hemi)
        to_surf = self.load(layer_name, stage=to_stage, hemi=to_hemi)
        from_vertices = from_surf.darrays[0].data
        to_vertices = to_surf.darrays[0].data

        # --- direct geometric mapping (default) ---
        def _geom_map(src, tgt):
            tree = cKDTree(tgt)
            _, idx = tree.query(src, k=1)
            return np.asarray(idx, dtype=int)

        # --- specific transitions ---
        if from_stage == 'converted' and to_stage == 'nodeep':
            # vertices were removed -> load JSON sidecar
            meta = self.load_meta(layer_name, stage='nodeep', hemi=from_hemi)
            removed = np.array(meta.get('deep_vertices_removed', []), dtype=int)
            keep_mask = np.ones(from_vertices.shape[0], dtype=bool)
            keep_mask[removed] = False
            idx = np.flatnonzero(keep_mask)

        elif from_stage == 'nodeep' and to_stage == 'combined':
            # hemisphere concatenation
            if from_hemi == 'lh':
                idx = np.arange(from_vertices.shape[0])
            elif from_hemi == 'rh':
                # offset by LH vertex count in combined
                lh_surf = self.load(layer_name, stage='nodeep', hemi='lh')
                idx = np.arange(from_vertices.shape[0]) + lh_surf.darrays[0].data.shape[0]
            else:
                raise ValueError("Specify from_hemi ('lh' or 'rh') when mapping from nodeep.")

        elif from_stage == 'combined' and to_stage == 'nodeep':
            # splitting combined back into hemis
            lh_surf = self.load(layer_name, stage='nodeep', hemi='lh')
            lh_n = lh_surf.darrays[0].data.shape[0]
            if to_hemi == 'lh':
                idx = np.arange(lh_n)
            elif to_hemi == 'rh':
                idx = np.arange(lh_n, to_vertices.shape[0])
            else:
                raise ValueError("Specify to_hemi ('lh' or 'rh') when mapping to nodeep.")

        elif (from_stage, to_stage) in [('combined', 'ds'),
                                        ('ds', 'combined')]:
            # downsampling or upsampling
            idx = _geom_map(from_vertices, to_vertices)

        else:
            # generic fallback
            idx = _geom_map(from_vertices, to_vertices)

        return idx

    # pylint: disable=R0912, R1702
    def validate(self, required_stages=('raw', 'converted', 'nodeep', 'combined', 'ds'),
                 hemis=('lh', 'rh'), orientations=None, fixed=None):
        """
        Validate the completeness and internal consistency of surface files across layers,
        hemispheres, and stages.

        This function checks that all expected surface files are present for the specified
        processing stages and verifies vertex count consistency across layers within each stage.
        For downsampled surfaces, optional validation of orientation and fixed-orientation variants
        is supported. The function raises descriptive errors for missing files or mismatched vertex
        counts, ensuring structural integrity of the laminar surface set before further processing.

        Parameters
        ----------
        required_stages : sequence of str, optional
            List of processing stages to validate (e.g., 'raw', 'converted', 'nodeep', 'combined',
            'ds'). Default includes all standard stages.
        hemis : sequence of {'lh', 'rh'}, optional
            Hemispheres to check for per-hemisphere stages (default: both).
        orientations : sequence of str or None, optional
            Orientation methods to validate for the downsampled stage (e.g., ['link_vector']).
            Ignored for other stages.
        fixed : bool or None, optional
            Whether to restrict validation to fixed-orientation variants. If None, both fixed and
            non-fixed variants are checked.

        Raises
        ------
        FileNotFoundError
            If any expected surface files are missing from the directory structure.
        ValueError
            If vertex counts are inconsistent across layers within a hemisphere or combined mesh.

        Notes
        -----
        - For 'raw', 'converted', and 'nodeep' stages, each hemisphere is validated separately.
        - For 'combined' surfaces, vertex counts are checked across layers for the merged
          hemispheres.
        - For 'ds' (downsampled) surfaces, each orientation and fixedness combination is validated
          independently.
        - This validation step is typically run at the end of `create` to
          ensure consistency before dipole orientation computation or laminar combination.
        """
        missing = []
        layer_names = self.get_layer_names()

        for stage in required_stages:
            layer_verts = {}

            # Orientation-aware validation only for downsampled stage
            if stage == 'ds' and orientations:
                fixed_states = [fixed] if fixed is not None else [True, False]
                for orientation in orientations:
                    for fx_state in fixed_states:
                        for layer_name in layer_names:
                            fname = self.get_mesh_path(
                                layer_name=layer_name,
                                stage=stage,
                                orientation=orientation,
                                fixed=fx_state,
                            )
                            if not os.path.exists(fname):
                                missing.append(fname)
                                continue
                            surf = self.load(
                                layer_name,
                                stage=stage,
                                orientation=orientation,
                                fixed=fx_state,
                            )
                            n_verts = surf.darrays[0].data.shape[0]
                            key = f"{orientation}_{'fixed' if fx_state else 'not_fixed'}"
                            if key not in layer_verts:
                                layer_verts[key] = n_verts
                            elif n_verts != layer_verts[key]:
                                raise ValueError(
                                    f"Vertex mismatch for '{orientation}' ({layer_name}): "
                                    f"expected {layer_verts[key]}, found {n_verts}"
                                )

            elif stage in ('raw', 'converted', 'nodeep'):
                for hemi in hemis:
                    for layer_name in layer_names:
                        fname = self.get_mesh_path(layer_name, stage=stage, hemi=hemi,
                                                   orientation=None, fixed=None)
                        if not os.path.exists(fname):
                            missing.append(fname)
                            continue
                        surf = self.load(layer_name, stage=stage, hemi=hemi)
                        n_verts = surf.darrays[0].data.shape[0]
                        if hemi not in layer_verts:
                            layer_verts[hemi] = n_verts
                        elif n_verts != layer_verts[hemi]:
                            raise ValueError(
                                f"Vertex mismatch in {layer_name} ({hemi}): "
                                f"expected {layer_verts[hemi]}, found {n_verts}"
                            )

            elif stage == 'combined':
                for layer_name in layer_names:
                    fname = self.get_mesh_path(layer_name, stage=stage, orientation=None,
                                               fixed=None)
                    if not os.path.exists(fname):
                        missing.append(fname)
                        continue
                    surf = self.load(layer_name, stage=stage)
                    n_verts = surf.darrays[0].data.shape[0]
                    if 'combined' not in layer_verts:
                        layer_verts['combined'] = n_verts
                    elif n_verts != layer_verts['combined']:
                        raise ValueError(
                            f"Vertex mismatch in {layer_name}: "
                            f"expected {layer_verts['combined']}, found {n_verts}"
                        )

        if missing:
            raise FileNotFoundError("Missing expected surface files:\n" + "\n".join(missing))

    def get_vertices_per_layer(self, orientation='link_vector', fixed=True):
        """
        Return the number of vertices per cortical layer in the downsampled multilayer mesh.

        This method loads the multilayer surface (spanning all cortical depths) and computes
        how many vertices correspond to a single layer by dividing the total vertex count
        by the number of layers in the surface set.

        Parameters
        ----------
        orientation : str, optional
            Dipole orientation model used in the laminar surface reconstruction
            (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Default is 'link_vector'.
        fixed : bool, optional
            Whether fixed dipole orientations were used across layers. Default is True.

        Returns
        -------
        n_vertices : int
            Number of vertices per cortical layer in the downsampled (stage='ds') mesh.
        """
        multilayer_mesh = self.load(
            layer_name=None,
            stage='ds',
            orientation=orientation,
            fixed=fixed
        )
        return int(multilayer_mesh.darrays[0].data.shape[0]/self.n_layers)

    # pylint: disable=R0912,R0915
    def compute_dipole_orientations(self, method, fixed=True):
        """
        Compute vertex-wise dipole orientation vectors for all cortical layers.

        This function estimates dipole orientations using one of several geometric methods applied
        to the laminar surface set. Depending on the method, orientations may be derived from
        vertex correspondence between pial and white surfaces or from local surface normals.
        Optionally, orientations can be fixed across layers, ensuring consistent directionality
        between homologous vertices.

        Parameters
        ----------
        method : {'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps'}
            Algorithm used to compute dipole orientations:
              - **'link_vector'** - vectors connecting pial and white vertices.
              - **'ds_surf_norm'** - surface normals computed from the downsampled surfaces.
              - **'orig_surf_norm'** - normals from original (combined) surfaces, mapped to the
                downsampled meshes.
              - **'cps'** - cortical patch statistics; averages of original-surface normals for
                vertices mapped to each downsampled vertex.
        fixed : bool, optional
            If True (default), the orientation vectors from the pial surface are reused for all
            layers, ensuring layer-consistent directionality. If False, orientations are computed
            separately for each layer.

        Returns
        -------
        orientations : np.ndarray
            Array of shape (n_layers, n_vertices, 3) containing normalized dipole orientation
            vectors.

        Raises
        ------
        ValueError
            If the pial and white surfaces have mismatched vertex counts when using the
            'link_vector' method.

        Notes
        -----
        - All orientation vectors are normalized to unit length.
        - For 'orig_surf_norm' and 'cps', geometric mapping between stages is computed using
          nearest-neighbour search.
        """
        orientations = None
        layer_names = self.get_layer_names()

        if method == 'link_vector':
            # Method: Use link vectors between pial and white surfaces as dipole orientations
            # Load downsampled pial and white surfaces
            pial_surf = self.load('pial', stage='ds')
            white_surf = self.load('white', stage='ds')

            # Extract vertices
            pial_vertices = pial_surf.darrays[0].data
            white_vertices = white_surf.darrays[0].data

            # Ensure same number of vertices in pial and white surfaces
            if pial_vertices.shape[0] != white_vertices.shape[0]:
                raise ValueError("Pial and white surfaces must have the same number of vertices")

            # Compute link vectors
            link_vectors = white_vertices - pial_vertices
            link_vectors = _normit(link_vectors)

            # Replicate link vectors for each layer
            orientations = np.tile(link_vectors, (len(layer_names), 1, 1))

        elif method == 'ds_surf_norm':
            # Method: Use normals of the downsampled surfaces
            orientations = []
            for l_idx, layer_name in enumerate(layer_names):
                if l_idx == 0 or not fixed:
                    surf = self.load(layer_name, stage='ds')
                    vtx_norms, _ = _vertex_normal_vectors(
                        surf.darrays[0].data,
                        surf.darrays[1].data,
                        unit=True
                    )
                orientations.append(vtx_norms)
            orientations = np.array(orientations)

        elif method == 'orig_surf_norm':
            # Method: Use normals of the original surfaces, mapped to downsampled surfaces
            orientations = []
            for l_idx, layer_name in enumerate(layer_names):
                if l_idx == 0 or not fixed:
                    orig_surf = self.load(layer_name, stage='combined')
                    orig_vert_idx = self.map_between_stages(layer_name, from_stage='ds',
                                                            to_stage='combined')

                    vtx_norms, _ = _vertex_normal_vectors(
                        orig_surf.darrays[0].data,
                        orig_surf.darrays[1].data,
                        unit=True
                    )
                orientations.append(vtx_norms[orig_vert_idx, :])
            orientations = np.array(orientations)

        elif method == 'cps':
            # Method: Use cortical patch statistics for normals
            orientations = []
            for l_idx, layer_name in enumerate(layer_names):
                if l_idx == 0 or not fixed:
                    orig_surf = self.load(layer_name, stage='combined')
                    ds_surf = self.load(layer_name, stage='ds')
                    ds_vert_idx = self.map_between_stages(layer_name, from_stage='combined',
                                                          to_stage='ds')

                    orig_vtx_norms, _ = _vertex_normal_vectors(
                        orig_surf.darrays[0].data,
                        orig_surf.darrays[1].data,
                        unit=True
                    )
                    vtx_norms, _ = _vertex_normal_vectors(
                        ds_surf.darrays[0].data,
                        ds_surf.darrays[1].data,
                        unit=True
                    )
                    for v_idx in range(vtx_norms.shape[0]):
                        orig_idxs = np.where(ds_vert_idx == v_idx)[0]
                        if len(orig_idxs):
                            vtx_norms[v_idx, :] = np.mean(orig_vtx_norms[orig_idxs, :], axis=0)
                    vtx_norms = _normit(vtx_norms)
                orientations.append(vtx_norms)
            orientations = np.array(orientations)

        return orientations

    def downsample(self, ds_factor):
        """
        Downsample all combined cortical surface meshes using the VTK decimation algorithm.

        This function applies geometric mesh reduction to each combined surface (including all
        laminar layers and the inflated mesh) using the `vtkDecimatePro` algorithm. The first
        surface in the sequence defines the downsampling vertex mapping, which is then applied to
        all other surfaces to ensure topological and vertex-wise correspondence across layers.

        Parameters
        ----------
        ds_factor : float
            Fraction of vertices to retain during decimation (e.g., 0.1 retains 10% of vertices).

        Notes
        -----
        - The vertex mapping from the first (reference) surface is reused for all other layers to
          maintain laminar alignment.
        - Downsampling removes redundant vertices while preserving overall cortical geometry.
        - If vertex normals are present in the input GIFTI files, they are resampled and stored in
          the output surfaces.
        - Metadata for each surface includes:
            * `'ds_factor'` - the applied decimation ratio.
            * `'ds_removed_vertices'` - indices of removed vertices in the original mesh.
        - Downsampled surfaces are saved under the `'ds'` stage within the laminar directory.

        See Also
        --------
        _iterative_downsample_single_surface : Performs per-surface iterative decimation.
        _fix_non_manifold_edges : Repairs topological defects introduced by downsampling.
        _create_surf_gifti : Creates a valid GIFTI surface from vertices and faces.

        Examples
        --------
        >>> surf_set = LayerSurfaceSet('sub-01', n_layers=11)
        >>> surf_set.downsample(0.1)
        """

        layer_names = self.get_layer_names()
        surfaces_to_process = copy.copy(layer_names)
        surfaces_to_process.append('inflated')

        primary_surf = self.load(surfaces_to_process[0], stage='combined')
        primary_meta = self.load_meta(surfaces_to_process[0], stage='combined')
        primary_meta['ds_factor'] = ds_factor
        ds_primary_surf = _iterative_downsample_single_surface(primary_surf, ds_factor=ds_factor)
        reduced_vertices = ds_primary_surf.darrays[0].data
        reduced_faces = ds_primary_surf.darrays[1].data

        # Find the original vertices closest to the downsampled vertices
        kdtree = cKDTree(primary_surf.darrays[0].data)
        # Calculate the percentage of vertices retained
        decim_orig_dist, orig_vert_idx = kdtree.query(reduced_vertices, k=1)
        orig_vert_idx = np.squeeze(orig_vert_idx)
        # enforce deterministic tie-break by sorting indices for identical distances
        orig_vert_idx = np.asarray(orig_vert_idx, dtype=int)

        removed_vertices = np.setdiff1d(np.arange(primary_surf.darrays[0].data.shape[0]),
                                        orig_vert_idx)
        primary_meta['ds_removed_vertices'] = removed_vertices.tolist()

        print(
            f"{(1 - np.mean(decim_orig_dist > 0)) * 100}% of the vertices in the decimated "
            f"surface belong to the original surface."
        )

        # Save the downsampled primary surface with normals
        self.save(ds_primary_surf, layer_names[0], stage='ds', meta=primary_meta)

        # Process other surfaces
        for i in range(1, len(surfaces_to_process)):
            surf = self.load(surfaces_to_process[i], stage='combined')
            surf_meta = self.load_meta(surfaces_to_process[i], stage='combined')
            surf_meta['ds_factor'] = ds_factor
            surf_meta['ds_removed_vertices'] = removed_vertices.tolist()

            reduced_normals = None
            if len(surf.darrays) > 2 and \
                    surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
                    surf.darrays[2].data.shape[0] == surf.darrays[0].data.shape[0]:
                reduced_normals = surf.darrays[2].data[orig_vert_idx]

            surf_verts = surf.darrays[0].data[orig_vert_idx, :]

            nonmani_vertices, nonmani_faces = _fix_non_manifold_edges(surf_verts, reduced_faces)

            ds_surf = _create_surf_gifti(nonmani_vertices, nonmani_faces, normals=reduced_normals)

            self.save(ds_surf, surfaces_to_process[i], stage='ds', meta=surf_meta)

    def get_cortical_thickness(self, stage='ds', hemi=None):
        """
        Compute vertex-wise cortical thickness from pial and white matter surfaces.

        This method loads the pial and white matter meshes at the specified processing stage
        and hemisphere, then computes cortical thickness as the Euclidean distance between
        corresponding vertices on the two surfaces. Vertex correspondence is maintained across
        layers, allowing accurate per-vertex thickness estimation.

        Parameters
        ----------
        stage : str, optional
            Surface processing stage to use (e.g., 'converted', 'nodeep', 'combined', 'ds').
            Default is 'ds'.
        hemi : {'lh', 'rh', None}, optional
            Hemisphere to process ('lh' or 'rh'). If None, the combined surface is used
            (default: None).

        Returns
        -------
        thickness : numpy.ndarray, shape (n_vertices,)
            Vertex-wise cortical thickness values (in millimeters).

        Notes
        -----
        - Thickness is computed as the Euclidean distance between corresponding pial and white
          matter vertices.
        - Requires vertex correspondence between surfaces, as guaranteed by `LayerSurfaceSet`.
        - Can be used to assess local cortical geometry or normalize laminar profiles.
        """
        pial_mesh = self.load(layer_name='pial', stage=stage, hemi=hemi)
        white_mesh = self.load(layer_name='white', stage=stage, hemi=hemi)
        vert_diff=pial_mesh.darrays[0].data - white_mesh.darrays[0].data
        thickness = np.sqrt(np.sum((vert_diff) ** 2, axis=-1))
        return thickness

    def get_distance_to_scalp(self, layer_name='pial', stage='ds', hemi=None):
        """
        Compute the minimum Euclidean distance from each cortical vertex to the scalp surface.

        This method loads the cortical surface (e.g., pial or white matter) and computes, for each
        vertex, the shortest distance to the nearest point on the scalp mesh. The scalp surface is
        expected to be stored as `origscalp_2562.surf.gii` in the subject's MRI directory.

        Parameters
        ----------
        layer_name : str, optional
            Cortical surface layer to use (e.g., 'pial', 'white', or fractional depth). Default is
            'pial'.
        stage : str, optional
            Processing stage of the surface mesh (e.g., 'converted', 'nodeep', 'combined', 'ds').
            Default is 'ds'.
        hemi : {'lh', 'rh', None}, optional
            Hemisphere to process ('lh' or 'rh'). If None, the combined surface is used (default:
            None).

        Returns
        -------
        distances : numpy.ndarray, shape (n_vertices,)
            Euclidean distance (in millimeters) from each cortical vertex to the nearest scalp
            vertex.

        Notes
        -----
        - The scalp surface is expected at: `<SUBJECTS_DIR>/<subject>/mri/origscalp_2562.surf.gii`.
        - Uses a KD-tree search for efficient nearest-neighbor computation.
        - Useful for evaluating cortical depth relative to scalp or for spatial normalization of
          MEG sensitivity profiles.
        """
        scalp_mesh_fname = os.path.join(self.subj_dir, 'mri', 'origscalp_2562.surf.gii')
        scalp_mesh = nib.load(scalp_mesh_fname)
        scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates

        layer_mesh = self.load(layer_name=layer_name, stage=stage, hemi=hemi)
        pial_ds_vertices = layer_mesh.darrays[0].data  # Vertex coordinates

        # Build a KD-tree for the scalp vertices
        tree = cKDTree(scalp_vertices)

        # Find the nearest neighbor in the scalp mesh for each vertex in the downsampled cortical
        # mesh
        distances, _ = tree.query(pial_ds_vertices)

        return distances

    def get_radiality_to_scalp(self, layer_name='pial', orientation='link_vector', fixed=True):
        """
        Compute the radiality of cortical dipole orientations relative to the scalp surface.

        This function quantifies the degree to which each cortical orientation vector
        (e.g., dipole or column direction) is aligned with the local scalp normal.
        The result is a vertex-wise cosine similarity (absolute dot product) between
        dipole orientation and scalp normal vectors, indicating radial alignment.

        Parameters
        ----------
        layer_name : str, optional
            Cortical surface layer to analyze (e.g., 'pial', 'white', or fractional depth).
            Default is 'pial'.
        orientation : str, optional
            Dipole orientation model used in the surface reconstruction
            (e.g., 'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps').
            Default is 'link_vector'.
        fixed : bool, optional
            Whether fixed dipole orientations were used during inversion. Default is True.

        Returns
        -------
        radiality : numpy.ndarray, shape (n_vertices,)
            Absolute cosine similarity between cortical dipole vectors and local scalp normals.
            Values range from 0 (tangential) to 1 (perfectly radial).

        Notes
        -----
        - Higher values indicate more radial orientations (dipoles pointing toward the scalp).
        - Scalp geometry is loaded from `<SUBJECTS_DIR>/<subject>/mri/origscalp_2562.surf.gii`.
        - Surface normals are estimated per face and averaged across connected vertices.
        - Useful for assessing source sensitivity, forward-model biases, or validating
          laminar orientation models.
        """
        scalp_mesh_fname = os.path.join(self.subj_dir, 'mri', 'origscalp_2562.surf.gii')
        scalp_mesh = nib.load(scalp_mesh_fname)
        scalp_vertices = scalp_mesh.darrays[1].data  # Vertex coordinates
        scalp_faces = scalp_mesh.darrays[0].data  # Face indices

        layer_mesh = self.load(
            layer_name=layer_name,
            stage='ds',
            orientation=orientation,
            fixed=fixed
        )
        layer_vertices = layer_mesh.darrays[0].data  # Vertex coordinates
        layer_orientations = layer_mesh.darrays[2].data  # Orientation vectors

        # Compute surface normals
        def compute_normals(vertices, faces):
            normals = np.zeros_like(vertices)
            for i in range(faces.shape[0]):
                v_0, v_1, v_2 = vertices[faces[i]]
                # Edge vectors
                edge1 = v_1 - v_0
                edge2 = v_2 - v_0
                # Cross product to get normal
                normal = np.cross(edge1, edge2)
                normal /= np.linalg.norm(normal)  # Normalize the normal
                normals[faces[i]] += normal
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
            return normals

        normals = compute_normals(scalp_vertices, scalp_faces)

        # Build a KD-tree for the scalp vertices
        tree = cKDTree(scalp_vertices)

        # Find the nearest neighbor in the scalp mesh for each vertex in the layer mesh
        _, indices = tree.query(layer_vertices)

        # Compute the dot product between the orientation and the normal vectors
        dot_products = np.abs(np.einsum('ij,ij->i', normals[indices], layer_orientations))
        return dot_products

    def interpolate_layer_data(self, layer_name, data, from_stage='ds', to_stage='combined',
                               k_neighbors=5):
        """
        Interpolate vertex-wise data from a downsampled layer mesh to the original high-resolution
        surface.

        This function projects data values defined on a downsampled surface (e.g., 'ds' stage) onto
        the corresponding vertices of the full-resolution mesh (e.g., 'combined' stage).
        Interpolation is performed using a nearest-neighbour scheme with optional local averaging
        over connected vertices.

        Parameters
        ----------
        layer_name : str
            Name of the cortical layer to interpolate (e.g., 'pial', 'white', '0.333').
        data : array-like, shape (n_vertices_ds,)
            Vertex-wise data associated with the downsampled mesh.
        from_stage : str, optional
            Source mesh stage for interpolation (default: 'ds').
        to_stage : str, optional
            Target mesh stage to which data are interpolated (default: 'combined').
        k_neighbors : int, optional
            Number of nearest neighbours to use for smoothing interpolation (default: 5).

        Returns
        -------
        interpolated_data : np.ndarray
            Array of interpolated vertex values in the target mesh space.

        Notes
        -----
        - The interpolation preserves the laminar correspondence between meshes by using vertex
          adjacency and spatial proximity.
        - This function is typically used to project laminar or orientation data back to the
          original-resolution FreeSurfer space after downsampling.
        - Internally calls `interpolate_data()` to perform weighted vertex interpolation.
        """
        orig_surf = self.load(layer_name, stage=to_stage)
        ds_surf = self.load(layer_name, stage=from_stage)
        adjacency = _mesh_adjacency(orig_surf.darrays[1].data)
        return interpolate_data(orig_surf, ds_surf, data, adjacency, k_neighbors)

    def get_bigbrain_layer_boundaries(self, subj_coord=None):
        """
        Map BigBrain proportional layer boundaries into the subject's downsampled surface space.

        This function retrieves the laminar boundary proportions defined in the BigBrain
        histological atlas and projects them into the subject's cortical geometry by mapping
        through fsaverage correspondence. It returns the proportional depth boundaries (0-1 range)
        of the six cortical layers for each vertex in the subject's surface space.

        Parameters
        ----------
        subj_coord : array-like or None, optional
            Optional array of vertex coordinates (n_vertices × 3) in the subject's native space.
            If None, mapping is performed for all vertices of the subject's pial surface.

        Returns
        -------
        layer_boundaries : np.ndarray
            Array of proportional cortical layer boundaries per vertex, shape (n_layers,
            n_vertices). If both hemispheres are processed, returns an array of shape (n_layers,
            n_total_vertices) with hemisphere stacking.

        Notes
        -----
        - Uses `convert_native_to_fsaverage` to identify corresponding fsaverage vertices for the
          subject.
        - Retrieves laminar proportion data from `big_brain_proportional_layer_boundaries()`.
        - The returned values represent depth fractions (0 at white matter, 1 at pial surface) that
          can be used for aligning laminar CSD or source estimates across subjects.
        """
        hemi, fsave_v_idx = convert_native_to_fsaverage(self, 'pial', subj_coord=subj_coord)
        bb_prop = big_brain_proportional_layer_boundaries()
        if isinstance(hemi, str):
            return bb_prop[hemi][:, fsave_v_idx]
        return np.stack([bb_prop[h][:, idx] for h, idx in zip(hemi, fsave_v_idx)], axis=1)


    def _create_layer_surfaces(self, hemispheres=('lh', 'rh'), n_jobs=-1):
        """
        Generate cortical layer surface meshes for the specified hemispheres.

        This function creates intermediate cortical surfaces between the white and pial boundaries
        based on the proportional thickness values defined in `self.layer_spacing`. If a given
        layer surface file does not exist, it is generated using FreeSurfer's `mris_expand` command
        applied to the white matter surface. Existing surfaces are reused without regeneration.

        Parameters
        ----------
        hemispheres : sequence of {'lh', 'rh'}, optional
            Hemispheres for which to generate layer surfaces. Default is ('lh', 'rh').
        n_jobs : int, optional
            Number of parallel jobs to use for surface generation. Default is -1 (use all available
            cores).

        Returns
        -------
        list of str
            List of generated or existing layer names, including 'pial', 'white', and intermediate
            fractional layers (e.g., ['pial', '0.900', '0.800', ..., 'white']).

        Notes
        -----
        - Uses `mris_expand` to generate intermediate surfaces when missing.
        - Layers are defined proportionally from white matter (0.0) to pial (1.0).
        - The function runs in parallel using joblib for efficiency.
        """

        def _create_layer(layer):
            if layer == 1:
                return 'pial'
            if layer == 0:
                return 'white'
            layer_name = f'{layer:.3f}'
            for hemi in hemispheres:
                wm_file = os.path.join(self.surf_dir, f'{hemi}.white')
                out_file = os.path.join(self.surf_dir, f'{hemi}.{layer_name}')
                if not os.path.exists(out_file):
                    cmd = ['mris_expand', '-thickness', wm_file, str(layer), out_file]
                    subprocess.run(cmd, check=True)
            return layer_name

        layer_names = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_create_layer)(layer) for layer in self.layer_spacing
        )
        return layer_names


    # pylint: disable=R0912,R0915
    def create(self, ds_factor=0.1, orientation='link_vector', fix_orientation=True, n_jobs=-1):
        """
        Postprocess and combine FreeSurfer cortical surface meshes for laminar analysis.

        This function reconstructs a complete laminar surface hierarchy for a given FreeSurfer
        subject. It generates intermediate cortical layers between the pial and white matter
        surfaces, converts all surfaces to GIFTI format, aligns them to scanner RAS coordinates,
        removes deep-cut vertices, combines hemispheres, performs mesh decimation, and computes
        dipole orientation vectors. The resulting downsampled, orientation-annotated multilayer
        surfaces are stored in:
        ``$SUBJECTS_DIR/<subj_id>/surf/laminar``

        Parameters
        ----------
        ds_factor : float, optional
            Fraction of vertices to retain during mesh decimation (default: 0.1).
        orientation : {'link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps'}, optional
            Method used to compute dipole orientations:
              - ``'link_vector'``: vectors connecting pial - white vertices.
              - ``'ds_surf_norm'``: surface normals from downsampled meshes.
              - ``'orig_surf_norm'``: surface normals from original combined meshes.
              - ``'cps'``: cortical patch statistics (mean local normals).
            Default is ``'link_vector'``.
        fix_orientation : bool, optional
            If True, enforces consistent orientation across layers by copying pial-layer normals
            to all other layers (default: True).
        n_jobs : int, optional
            Number of parallel jobs used when generating intermediate surfaces (default: -1 for all
            cores).

        Raises
        ------
        EnvironmentError
            If required FreeSurfer binaries (``mris_convert``, ``mris_inflate``, ``mri_info``)
            are not found in the system PATH.
        FileNotFoundError
            If the subject directory or expected FreeSurfer files are missing.
        ValueError
            If validation fails due to vertex count mismatches across layers or hemispheres.

        Notes
        -----
        - Requires that FreeSurfer's ``$SUBJECTS_DIR`` environment variable is set and
          ``recon-all`` has completed successfully for ``subj_id``.
        - Surface metadata JSON files record coordinate transformations, removed vertices,
          downsampling ratios, and orientation method parameters.
        - Downsampling and orientation computation are applied identically across all layers
          to preserve laminar alignment.
        """

        # --- Check that required FreeSurfer binaries are available ---
        check_freesurfer_setup()

        # Convert MRI to nii
        orig_mgz = os.path.join(self.subj_dir, 'mri', 'orig.mgz')
        orig_nii = os.path.join(self.subj_dir, 'mri', 'orig.nii')
        img = nib.load(orig_mgz)
        nib.save(img, orig_nii)

        hemispheres = ['lh', 'rh']

        out_dir = self.laminar_surf_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Create intermediate surfaces if needed
        layer_names = self._create_layer_surfaces(hemispheres=hemispheres, n_jobs=n_jobs)

        surfaces_to_process = copy.copy(layer_names)
        surfaces_to_process.append('inflated')

        # Convert to gifti
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                # Construct the original and new file names
                orig_name = os.path.join(self.surf_dir, f'{hemi}.{surface_name}')
                new_name = os.path.join(out_dir, f'{hemi}.{surface_name}.raw.gii')

                # Convert the surface file to Gifti format
                subprocess.run(['mris_convert', orig_name, new_name], check=True)

        def _mat(flag):
            out = subprocess.check_output(['mri_info', flag, orig_mgz]).decode().split()
            return np.array([float(x) for x in out]).reshape(4, 4)

        t_orig = _mat('--vox2ras-tkr')  # vox -> tkRAS (mm)
        n_orig = _mat('--vox2ras')  # vox -> scanner RAS (mm)
        inv_t_orig = np.linalg.inv(t_orig)

        def tkras_to_scanner_ras(coords_mm):
            # coords_mm: (N,3) tkRAS
            xyz1 = np.c_[coords_mm, np.ones((coords_mm.shape[0], 1))]
            out = (n_orig @ (inv_t_orig @ xyz1.T)).T[:, :3]
            return out

        meta = {
            't_orig': t_orig.tolist(),
            'n_orig': n_orig.tolist()
        }

        # Convert from tkRAS to scanner coordinates
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                # Load the Gifti file
                surf_g = self.load(surface_name, stage='raw', hemi=hemi)

                # Set transformation matrix to identity
                surf_g.affine = np.eye(4)

                # Convert from tkRAS to scanner coordinates
                for data_array in surf_g.darrays:
                    if data_array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                        data_array.data = tkras_to_scanner_ras(data_array.data)
                self.save(surf_g, surface_name, stage='converted', hemi=hemi, meta=meta)

        # Remove vertices created by cutting the hemispheres
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                surf_g = self.load(surface_name, stage='converted', hemi=hemi)
                surf_meta = copy.copy(meta)

                n_vertices = 0
                for data_array in surf_g.darrays:
                    if data_array.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                        n_vertices = data_array.data.shape[0]

                annotation = os.path.join(self.subj_dir, 'label', f'{hemi}.aparc.annot')
                label, _, names = nib.freesurfer.read_annot(annotation)

                # Remove vertices created by cutting the hemispheres
                vertices_to_remove = []
                for vtx in range(n_vertices):
                    region_name = names[label[vtx]]
                    if isinstance(region_name, bytes):
                        region_name = region_name.decode('utf-8', errors='ignore')
                    if label[vtx] <= 0 or region_name.lower() == 'unknown':
                        vertices_to_remove.append(vtx)
                surf_g = _remove_vertices(surf_g, np.array(vertices_to_remove))

                # Save the modified Gifti file
                surf_meta['deep_vertices_removed'] = vertices_to_remove
                self.save(surf_g, surface_name, stage='nodeep', hemi=hemi, meta=surf_meta)

        req_stages = ['raw', 'converted', 'nodeep']
        self.validate(required_stages=req_stages, hemis=('lh', 'rh'))

        # Combine hemispheres
        for surface_name in surfaces_to_process:
            # Load left and right hemisphere surfaces
            l_hemi = self.load(surface_name, stage='nodeep', hemi='lh')
            l_hemi_meta = self.load_meta(surface_name, stage='nodeep', hemi='lh')
            r_hemi = self.load(surface_name, stage='nodeep', hemi='rh')
            r_hemi_meta = self.load_meta(surface_name, stage='nodeep', hemi='rh')

            if surface_name == 'inflated':
                lh_width = np.max(l_hemi.darrays[0].data[:, 0]) - \
                           np.min(l_hemi.darrays[0].data[:, 0])
                shift_amount = np.max(l_hemi.darrays[0].data[:, 0]) + (.5 * lh_width)
                r_hemi.darrays[0].data[:, 0] = r_hemi.darrays[0].data[:, 0] + shift_amount

            # Combine the surfaces
            combined = _concatenate_surfaces([l_hemi, r_hemi])

            surf_meta = copy.copy(meta)
            surf_meta['lh_deep_vertices_removed'] = l_hemi_meta['deep_vertices_removed']
            surf_meta['rh_deep_vertices_removed'] = r_hemi_meta['deep_vertices_removed']

            self.save(combined, surface_name, stage='combined', meta=surf_meta)

        # Downsample surfaces at the same time
        self.downsample(ds_factor)
        meta['ds_factor'] = ds_factor

        # Compute dipole orientations
        orientations = self.compute_dipole_orientations(
            orientation,
            fixed=fix_orientation
        )
        meta['orientation'] = orientation
        meta['fix_orientation'] = fix_orientation

        for l_idx, layer_name in enumerate(layer_names):
            surf = self.load(layer_name, stage='ds')
            surf_meta = self.load_meta(layer_name, stage='ds')
            surf_meta['orientation'] = orientation
            surf_meta['fix_orientation'] = fix_orientation

            # Set these vectors as the orientations for the downsampled surface
            intent_code=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']
            ori_array = nib.gifti.GiftiDataArray(data=orientations[l_idx, :, :],
                                                 intent=intent_code)
            surf.add_gifti_data_array(ori_array)

            # Save the modified downsampled surface with orientation vectors
            self.save(surf, layer_name, stage='ds', orientation=orientation, fixed=fix_orientation,
                      meta=surf_meta)

        # Combine layers
        all_surfs = []
        for layer_name in layer_names:
            surf = self.load(layer_name, stage='ds', orientation=orientation, fixed=fix_orientation)
            surf_meta = self.load_meta(layer_name, stage='ds', orientation=orientation,
                                           fixed=fix_orientation)
            meta[f'lh_{layer_name}_deep_vertices_removed'] = surf_meta['lh_deep_vertices_removed']
            meta[f'rh_{layer_name}_deep_vertices_removed'] = surf_meta['rh_deep_vertices_removed']
            meta[f'{layer_name}_ds_removed_vertices'] = surf_meta['ds_removed_vertices']
            all_surfs.append(surf)

        combined = _concatenate_surfaces(all_surfs)
        meta['n_layers'] = self.n_layers

        self.save(combined, f'multilayer.{self.n_layers}', stage='ds', orientation=orientation,
                  fixed=fix_orientation, meta=meta)

        self.validate(
            required_stages=(
                'combined',
                'ds'
            ),
            orientations=[orientation],
            fixed=fix_orientation
        )


def convert_fsaverage_to_native(surf_set, layer_name, hemi, vert_idx=None):
    """
    Map vertex indices from fsaverage spherical registration space to a subject's native,
    downsampled surface space.

    This function maps vertices defined on the fsaverage template to their corresponding
    locations on an individual subject's cortical surface. It first identifies the subject's
    corresponding vertex in spherical registration space (`?h.sphere.reg`) via nearest-neighbour
    search, then maps those vertices to the subject's downsampled surface for use in laminar or
    source-space analyses.

    Parameters
    ----------
    surf_set : LayerSurfaceSet
        Subject's surface set instance containing paths and metadata for all cortical layers.
    layer_name : str
        Name of the surface layer to use for mapping (e.g., 'pial', 'white', or a fractional layer).
    hemi : {'lh', 'rh'}
        Hemisphere to convert.
    vert_idx : int | array-like of int | None, optional
        Vertex index or array of vertex indices on the fsaverage surface.
        If None, all fsaverage vertices are mapped.

    Returns
    -------
    subj_v_idx : int or np.ndarray
        Corresponding vertex index (or array of indices) on the subject's downsampled surface.

    Notes
    -----
    - The mapping is performed in two stages:
        1. **Spherical registration mapping:** fsaverage vertices are mapped to the subject's
           `?h.sphere.reg` using nearest-neighbour search in spherical space.
        2. **Downsampling alignment:** the subject's full-resolution surface vertices are
           mapped onto the downsampled mesh via spatial proximity.
    - Assumes that the downsampled surface is a vertex subset of the full-resolution mesh.
    - This provides a lightweight geometric approximation to FreeSurfer's surface morph,
      suitable for vertex-level correspondence in laminar analyses.

    Examples
    --------
    >>> subj_v_idx = convert_fsaverage_to_native(surf_set, 'pial', 'lh', vert_idx=[100, 200, 300])
    >>> subj_v_idx.shape
    (3,)
    """
    subj_dir = os.path.join(surf_set.subjects_dir, surf_set.subj_id)

    # --- Load spherical registration surfaces ---
    fsaverage_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(surf_set.subjects_dir, 'fsaverage', 'surf', f'{hemi}.sphere.reg')
    )
    subj_sphere_vertices, _ = nib.freesurfer.read_geometry(
        os.path.join(subj_dir, 'surf', f'{hemi}.sphere.reg')
    )

    # --- Default to all vertices if none specified ---
    if vert_idx is None:
        vert_idx = np.arange(fsaverage_sphere_vertices.shape[0])
    vert_idx = np.atleast_1d(vert_idx)

    # --- Map fsaverage vertices to subject's spherical registration space ---
    kdtree = cKDTree(subj_sphere_vertices)
    _, subj_v_idx_local = kdtree.query(fsaverage_sphere_vertices[vert_idx, :], k=1)
    subj_v_idx_local = np.atleast_1d(np.squeeze(subj_v_idx_local))
    # enforce deterministic tie-break by sorting indices for identical distances
    subj_v_idx_local = np.asarray(subj_v_idx_local, dtype=int)

    # --- Load full-resolution and downsampled surfaces ---
    subj_ds = surf_set.load(layer_name, stage='ds')
    ds_vertices = subj_ds.darrays[0].data

    subj_fr = surf_set.load(layer_name, stage='converted', hemi=hemi)
    fr_vertices = subj_fr.darrays[0].data

    # --- Map full-resolution vertices to downsampled vertices ---
    ds_kdtree = cKDTree(ds_vertices)
    _, v_idx = ds_kdtree.query(fr_vertices[subj_v_idx_local, :], k=1)
    v_idx = np.atleast_1d(np.squeeze(v_idx))
    # enforce deterministic tie-break by sorting indices for identical distances
    v_idx = np.asarray(v_idx, dtype=int)

    # --- Return scalar if input was scalar ---
    if np.ndim(vert_idx) == 0 or (len(vert_idx) == 1 and np.isscalar(vert_idx[0])):
        return int(v_idx[0])
    return v_idx


def convert_native_to_fsaverage(surf_set, layer_name, subj_coord=None):
    """
    Map coordinates from a subject's native surface space to fsaverage spherical registration space.

    This function maps vertices or coordinates defined in a subject's native surface (typically
    downsampled pial or white matter surfaces) to their corresponding vertices on the fsaverage
    template. It uses the spherical registration surfaces (`?h.sphere.reg`) to establish vertex-
    level correspondence via nearest-neighbour matching in spherical space.

    Parameters
    ----------
    surf_set : LayerSurfaceSet
        Subject''s surface set instance containing paths and metadata for cortical layers.
    layer_name : str
        Surface layer to use for mapping (e.g., 'pial', 'white', or a fractional layer).
    subj_coord : array-like of shape (3,) or (N, 3), optional
        Vertex coordinate(s) in the subject's downsampled surface space.
        If None, all downsampled vertices are mapped.

    Returns
    -------
    hemis : str or list of str
        Hemisphere label(s) for each vertex ('lh' or 'rh').
    fsave_v_idx : int or list of int
        Corresponding vertex index (or list of indices) on the fsaverage spherical surface.

    Notes
    -----
    The mapping proceeds in three stages:
      1. **Downsampled -> full-resolution mapping:** Each downsampled vertex is matched to the
         nearest vertex in the subject's full-resolution left and right hemisphere meshes.
      2. **Hemisphere assignment:** Vertices are assigned to the hemisphere with the smallest
         Euclidean distance.
      3. **Spherical registration mapping:** The corresponding vertex on the subject's
         `?h.sphere.reg` surface is matched to fsaverage using nearest-neighbour search in
         spherical space.

    This procedure approximates FreeSurfer's spherical morph alignment geometrically and is
    suitable for mapping subject-specific data into fsaverage space for group-level laminar or
    source-space analyses.

    Examples
    --------
    >>> hemis, fsave_idx = convert_native_to_fsaverage(surf_set, 'pial')
    >>> len(fsave_idx)
    81924
    >>> hemis[0], fsave_idx[0]
    ('lh', 10234)
    """
    fs_subjects_dir = surf_set.subjects_dir
    subj_dir = os.path.join(fs_subjects_dir, surf_set.subj_id)

    # --- Prepare downsampled vertex coordinates ---
    if subj_coord is None:
        ds_surf = surf_set.load(layer_name, stage='ds')
        ds_vertices = ds_surf.darrays[0].data
    else:
        ds_vertices = np.atleast_2d(subj_coord)

    # --- Load full-resolution surfaces ---
    subj_fr_lh = surf_set.load(layer_name, stage='converted', hemi='lh')
    subj_fr_rh = surf_set.load(layer_name, stage='converted', hemi='rh')
    lh_vertices = subj_fr_lh.darrays[0].data
    rh_vertices = subj_fr_rh.darrays[0].data

    # --- Map downsampled vertices to full-resolution hemispheres ---
    lh_kdtree = cKDTree(lh_vertices)
    rh_kdtree = cKDTree(rh_vertices)

    lh_dists, lh_idx = lh_kdtree.query(ds_vertices, k=1)
    rh_dists, rh_idx = rh_kdtree.query(ds_vertices, k=1)
    # enforce deterministic tie-break by sorting indices for identical distances
    lh_idx = np.asarray(lh_idx, dtype=int)
    rh_idx = np.asarray(rh_idx, dtype=int)

    hemis = np.where(lh_dists < rh_dists, 'lh', 'rh')
    subj_pial_idx = np.where(lh_dists < rh_dists, lh_idx, rh_idx)

    # --- Load spherical registration surfaces ---
    fsavg_lh_sphere, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'lh.sphere.reg')
    )
    fsavg_rh_sphere, _ = nib.freesurfer.read_geometry(
        os.path.join(fs_subjects_dir, 'fsaverage', 'surf', 'rh.sphere.reg')
    )

    subj_lh_sphere, _ = nib.freesurfer.read_geometry(
        os.path.join(subj_dir, 'surf', 'lh.sphere.reg')
    )
    subj_rh_sphere, _ = nib.freesurfer.read_geometry(
        os.path.join(subj_dir, 'surf', 'rh.sphere.reg')
    )

    fs_lh_tree = cKDTree(fsavg_lh_sphere)
    fs_rh_tree = cKDTree(fsavg_rh_sphere)

    # --- Map subject vertices to fsaverage ---
    subj_sphere_coords = np.array([
        subj_lh_sphere[idx] if hemi == 'lh' else subj_rh_sphere[idx]
        for hemi, idx in zip(hemis, subj_pial_idx)
    ])

    fsave_v_idx = np.array([
        fs_lh_tree.query(coord, k=1)[1] if hemi == 'lh'
        else fs_rh_tree.query(coord, k=1)[1]
        for hemi, coord in zip(hemis, subj_sphere_coords)
    ])
    fsave_v_idx = np.asarray(fsave_v_idx, dtype=int)

    # --- Return scalar if single coordinate provided ---
    if subj_coord is not None and len(ds_vertices) == 1:
        return hemis[0], int(fsave_v_idx[0])
    return hemis.tolist(), fsave_v_idx.tolist()


def interpolate_data(original_mesh, downsampled_mesh, downsampled_data, adjacency_matrix=None,
                     k_neighbors=5):
    """
    Interpolate vertex-wise data from a downsampled mesh back to the original high-resolution mesh.

    This function reconstructs vertex data on the full-resolution surface by weighted averaging
    of the *k* nearest downsampled vertices in Euclidean space. It then performs a refinement pass
    for vertices that were part of the downsampled mesh, averaging their values with adjacent
    vertices to enforce local smoothness.

    Parameters
    ----------
    original_mesh : nibabel.gifti.GiftiImage
        Original high-resolution cortical mesh (GIFTI format).
    downsampled_mesh : nibabel.gifti.GiftiImage
        Downsampled version of the mesh corresponding to the provided data.
    downsampled_data : np.ndarray
        Vertex-wise data array from the downsampled mesh.
    adjacency_matrix : scipy.sparse matrix, optional
        Vertex adjacency matrix for the original mesh. If None, it is computed internally.
    k_neighbors : int, optional
        Number of nearest downsampled vertices used for interpolation (default: 5).

    Returns
    -------
    vertex_data : np.ndarray
        Interpolated vertex-wise data for the original high-resolution mesh.

    Notes
    -----
    - The first interpolation pass uses inverse-distance weighting based on *k*-nearest neighbors
      in Euclidean space between meshes.
    - The second pass refines vertices present in the downsampled mesh using weighted neighborhood
      averaging based on surface adjacency.
    - This method preserves spatial smoothness while minimizing interpolation bias near downsampled
      vertices.

    Examples
    --------
    >>> vertex_data = interpolate_data(orig_gii, ds_gii, ds_data, k_neighbors=5)
    >>> vertex_data.shape
    (163842,)
    """
    original_vertices = original_mesh.darrays[0].data
    downsampled_vertices = downsampled_mesh.darrays[0].data

    # Build a KD-tree for the downsampled mesh
    tree = cKDTree(downsampled_vertices)

    # Find the k nearest downsampled vertices for each original vertex
    distances, indices = tree.query(original_vertices, k=k_neighbors)
    indices = np.squeeze(indices)
    indices = np.asarray(indices, dtype=int)

    # Initialize interpolated data array
    vertex_data = np.full(len(original_vertices), np.nan)

    # Compute weighted interpolation from k-nearest downsampled vertices
    valid_mask = indices[:, 0] < len(downsampled_data)  # Ensure indices are within bounds
    for i in np.where(valid_mask)[0]:
        valid_indices = indices[i, :]
        valid_distances = distances[i, :]

        # Avoid division by zero (replace zero distances with a small number)
        valid_distances[valid_distances == 0] = 1e-6
        weights = 1 / valid_distances
        weights /= weights.sum()  # Normalize weights

        vertex_data[i] = np.dot(weights, downsampled_data[valid_indices])

    # Second pass: Refine values for vertices that were in the downsampled mesh
    if adjacency_matrix is None:
        adjacency_matrix = _mesh_adjacency(original_mesh.darrays[1].data)

    downsampled_vertex_mask = np.isin(np.arange(len(original_vertices)), indices[:, 0])

    for i in np.where(downsampled_vertex_mask)[0]:
        neighbors = adjacency_matrix[i].nonzero()[1]
        if len(neighbors) > 0:
            distances = np.linalg.norm(original_vertices[neighbors] - original_vertices[i], axis=1)
            distances[distances == 0] = 1e-6  # Avoid division by zero
            weights = 1 / distances
            weights /= weights.sum()  # Normalize weights

            vertex_data[i] = np.dot(weights, vertex_data[neighbors])

    return vertex_data




# -------------------------------------------------------------------------
# Internal mesh utilities
# -------------------------------------------------------------------------

def _split_connected_components(faces, vertices):
    """
    Split a surface mesh into its connected components based on face adjacency.

    This function identifies and separates topologically distinct mesh components by
    recursively grouping faces that share at least one vertex. Each resulting component
    is returned as a separate set of faces and vertices.

    Parameters
    ----------
    faces : np.ndarray, shape (N, 3)
        Triangular face array, where each row contains vertex indices into `vertices`.
    vertices : np.ndarray, shape (M, 3)
        Vertex coordinate array corresponding to the face indices.

    Returns
    -------
    fv_out : list of dict
        A list of connected components. Each element is a dictionary with:
          - `'faces'`: (K, 3) array of faces within the component.
          - `'vertices'`: (L, 3) array of vertices belonging to the component.

    Notes
    -----
    - Vertices are reindexed within each component to maintain local face-vertex consistency.
    - Duplicate vertices at identical coordinates are not merged unless they share face indices.
    - This function assumes triangular faces and shared-vertex connectivity.

    Examples
    --------
    >>> parts = split_fv(faces, vertices)
    >>> len(parts)
    2
    >>> parts[0]['faces'].shape, parts[1]['vertices'].shape
    ((1024, 3), (500, 3))
    """

    num_faces = faces.shape[0]
    f_sets = np.zeros(num_faces, dtype=np.uint32)
    current_set = 0

    while np.any(f_sets == 0):
        current_set += 1
        next_avail_face = np.where(f_sets == 0)[0][0]
        open_vertices = faces[next_avail_face]

        while open_vertices.size:
            avail_face_inds = np.where(f_sets == 0)[0]
            is_member = np.isin(faces[avail_face_inds], open_vertices)
            avail_face_sub = np.where(np.any(is_member, axis=1))[0]
            f_sets[avail_face_inds[avail_face_sub]] = current_set
            open_vertices = np.unique(faces[avail_face_inds[avail_face_sub]])

    fv_out = []
    for set_num in range(1, current_set + 1):
        set_f = faces[f_sets == set_num]
        unique_vertices, new_vertex_indices = np.unique(set_f, return_inverse=True)
        fv_out.append({
            'faces': new_vertex_indices.reshape(set_f.shape),
            'vertices': vertices[unique_vertices]
        })

    return fv_out


def _mesh_adjacency(faces):
    """
    Compute a vertex adjacency matrix from a triangular surface mesh.

    This function derives the vertex-vertex connectivity structure of a mesh by
    identifying shared edges among faces. The resulting sparse matrix can be used
    for neighborhood-based computations such as smoothing, interpolation, or graph-based
    traversal.

    Parameters
    ----------
    faces : np.ndarray, shape (F, 3)
        Array of triangular faces, where each row contains vertex indices.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (V, V)
        Binary sparse adjacency matrix, where entry (i, j) = 1 indicates that
        vertices *i* and *j* share an edge. The matrix is symmetric.

    Notes
    -----
    - The number of vertices *V* is inferred from the maximum index in `faces` + 1.
    - Duplicate edges are merged automatically by sparse matrix construction.
    - The diagonal of the matrix is zero (no self-connections).

    Examples
    --------
    >>> adj = _mesh_adjacency(faces)
    >>> adj.shape
    (10242, 10242)
    >>> adj.nnz  # number of edges * 2 (since symmetric)
    61440
    """

    faces = np.asarray(faces, dtype=int)
    n_vertices = np.max(faces) + 1  # Assuming max vertex index represents the number of vertices

    # Flatten the indices to create row and column indices for the adjacency matrix
    row_indices = np.hstack([faces[:, 0], faces[:, 0], faces[:, 1],
                             faces[:, 1], faces[:, 2], faces[:, 2]])
    col_indices = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0],
                             faces[:, 2], faces[:, 0], faces[:, 1]])

    # Create a sparse matrix from row and column indices
    adjacency = csr_matrix(
        (np.ones_like(row_indices), (row_indices, col_indices)),
        shape=(n_vertices, n_vertices)
    )

    # Ensure the adjacency matrix is binary
    adjacency = (adjacency > 0).astype(int)

    return adjacency


def _normit(vectors):
    """
    Normalize an array of vectors to unit length.

    Each input vector (row) is divided by its Euclidean norm. Vectors with near-zero
    magnitude (below machine epsilon) are left unchanged to prevent numerical
    instability.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, 3)
        Array of N vectors to normalize.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Array of normalized vectors, each with unit Euclidean length.

    Notes
    -----
    - Norms smaller than machine epsilon are set to 1 before division to avoid
      division-by-zero errors.
    - The operation is performed row-wise.

    Examples
    --------
    >>> v = np.array([[3, 0, 0], [0, 4, 0]])
    >>> _normit(v)
    array([[1., 0., 0.],
           [0., 1., 0.]])
    """

    norm_n = np.sqrt(np.sum(vectors ** 2, axis=1))
    norm_n[norm_n < np.finfo(float).eps] = 1
    return vectors / norm_n[:, np.newaxis]


def _vertex_normal_vectors(vertices, faces, unit=False):
    """
    Compute per-vertex and per-face normal vectors for a triangular surface mesh.

    Face normals are computed as the cross product of two edges of each triangle,
    and vertex normals are estimated by summing the normals of adjacent faces.
    The overall orientation is corrected to ensure outward-facing consistency
    based on the mean dot product with centered vertex coordinates.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Array of vertex coordinates.
    faces : np.ndarray, shape (F, 3)
        Array of triangular faces, each row containing vertex indices.
    unit : bool, optional
        If True, return unit-length (normalized) normals. Default is False.

    Returns
    -------
    vertex_normal : np.ndarray, shape (V, 3)
        Normal vectors at each vertex.
    face_normal : np.ndarray, shape (F, 3)
        Normal vectors for each triangular face.

    Notes
    -----
    - Vertex normals are computed by summing the normals of adjacent faces.
    - The sign of normals is flipped if the majority point inward relative
      to the mesh centroid.
    - When `unit=True`, both vertex and face normals are normalized to unit length.

    Examples
    --------
    >>> v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> f = np.array([[0, 1, 2]])
    >>> vn, fn = _vertex_normal_vectors(v, f, unit=True)
    >>> fn
    array([[0., 0., 1.]])
    """
    face_normal = np.cross(
        vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
        vertices[faces[:, 2], :] - vertices[faces[:, 0], :]
    )
    face_normal = _normit(face_normal)

    vertex_normal = np.zeros_like(vertices)
    for i in range(len(faces)):
        for j in range(3):
            vertex_normal[faces[i, j], :] += face_normal[i, :]

    centered_vertices = vertices - np.mean(vertices, axis=0)
    if np.count_nonzero(np.sign(np.sum(centered_vertices * vertex_normal, axis=1))) > \
            len(centered_vertices) / 2:
        vertex_normal = -vertex_normal
        face_normal = -face_normal

    if unit:
        vertex_normal = _normit(vertex_normal)
        face_normal = _normit(face_normal)

    return vertex_normal, face_normal


def _create_surf_gifti(vertices, faces, normals=None):
    """
    Construct a GIFTI surface object from vertex, face, and optional normal data.

    This function builds a `nibabel.gifti.GiftiImage` representing a 3D surface mesh.
    Vertices and faces are required inputs; vertex normals are optional. Arrays are
    cast to appropriate datatypes (`float32` for coordinates and `int32` for faces)
    before being stored as GIFTI data arrays with the correct NIfTI intents.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Vertex coordinates in 3D space.
    faces : np.ndarray, shape (F, 3)
        Triangular face definitions, with each row containing vertex indices.
    normals : np.ndarray, shape (V, 3), optional
        Vertex normal vectors. If provided, stored with intent `'NIFTI_INTENT_VECTOR'`.

    Returns
    -------
    new_gifti : nibabel.gifti.GiftiImage
        GIFTI object containing the provided vertices, faces, and optional normals.

    Notes
    -----
    - Vertices and normals are stored as `float32`; faces are stored as `int32`.
    - The output follows FreeSurfer and HCP conventions for mesh geometry.
    - This function does not perform validation of mesh topology or normal orientation.

    Examples
    --------
    >>> v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> f = np.array([[0, 1, 2]])
    >>> gii = _create_surf_gifti(v, f)
    >>> isinstance(gii, nib.gifti.GiftiImage)
    True
    """

    # Create new gifti object
    new_gifti = nib.gifti.GiftiImage()

    # Cast vertices and faces to the appropriate data types
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)

    # Add the vertices and faces to the gifti object
    new_gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=vertices,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
        )
    )
    new_gifti.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data=faces,
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
        )
    )

    # If normals are provided and not empty, cast them to float32 and add them to the Gifti image
    if normals is not None:
        normals = np.array(normals).astype(np.float32)
        new_gifti.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                data=normals,
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']
            ))

    return new_gifti


def _remove_unconnected_vertices(gifti_surf):
    """
    Remove vertices that are not referenced by any face from a GIFTI surface.

    This function identifies vertices that are not part of any triangle in the
    mesh and removes them, along with updating the face indices accordingly.
    The cleaned surface is returned as a new `nibabel.gifti.GiftiImage`.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        Input GIFTI surface containing vertex and face data arrays.

    Returns
    -------
    cleaned_gifti_surf : nibabel.gifti.GiftiImage
        Surface with unconnected (or isolated) vertices removed.

    Notes
    -----
    - Vertices that do not appear in any face are considered unconnected.
    - Internally uses `_remove_vertices` to rebuild the surface after removal.
    - The relative geometry and face connectivity of the remaining mesh are preserved.

    Examples
    --------
    >>> gii = _create_surf_gifti(
    ...     np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]]),
    ...     np.array([[0, 1, 2]])
    ... )
    >>> cleaned = _remove_unconnected_vertices(gii)
    >>> cleaned.darrays[0].data.shape[0]
    3
    """

    # Get the pointset (vertices) and triangle array (faces) from the Gifti surface
    vertices = gifti_surf.darrays[0].data
    faces = gifti_surf.darrays[1].data

    # Find all unique vertex indices that are used in faces
    connected_vertices = np.unique(faces.flatten())

    # Determine which vertices are not connected to any faces
    all_vertices = np.arange(vertices.shape[0])
    unconnected_vertices = np.setdiff1d(all_vertices, connected_vertices)

    # Remove unconnected vertices using the provided remove_vertices function
    cleaned_gifti_surf = _remove_vertices(gifti_surf, unconnected_vertices)

    return cleaned_gifti_surf


def _remove_vertices(gifti_surf, vertices_to_remove):
    """
    Remove specified vertices from a GIFTI surface and update face connectivity.

    This function removes a set of vertices from a surface mesh and reindexes all faces
    to maintain valid connectivity among the remaining vertices. If vertex normals are
    present, they are filtered accordingly. The returned surface preserves topology for
    the remaining connected vertices.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        Input GIFTI surface containing vertex and face data arrays.
    vertices_to_remove : array_like
        Indices of vertices to remove (0-based). Vertices not listed are retained.

    Returns
    -------
    new_gifti : nibabel.gifti.GiftiImage
        Surface with specified vertices removed and face indices reindexed.

    Notes
    -----
    - Faces referencing any removed vertex are discarded.
    - If normals are present (`NIFTI_INTENT_VECTOR`), they are filtered to match the
      remaining vertices.
    - The operation is non-destructive: a new `GiftiImage` is returned, leaving the
      input object unchanged.

    Examples
    --------
    >>> v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]])
    >>> f = np.array([[0, 1, 2], [1, 2, 3]])
    >>> gii = _create_surf_gifti(v, f)
    >>> new_gii = _remove_vertices(gii, [3])
    >>> new_gii.darrays[0].data.shape[0]
    3
    """

    # Extract vertices and faces from the gifti object
    vertices_data = [da for da in gifti_surf.darrays
                     if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']][0]
    faces_data = [da for da in gifti_surf.darrays
                  if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']][0]

    vertices = vertices_data.data
    faces = faces_data.data

    # Determine vertices to keep
    vertices_to_keep = np.setdiff1d(np.arange(vertices.shape[0]), vertices_to_remove)

    # Create new array of vertices
    new_vertices = vertices[vertices_to_keep, :]

    # Find which faces to keep - ones that point to kept vertices
    if faces.shape[0] > 0:
        face_x = np.isin(faces[:, 0], vertices_to_keep)
        face_y = np.isin(faces[:, 1], vertices_to_keep)
        face_z = np.isin(faces[:, 2], vertices_to_keep)
        faces_to_keep = np.where(face_x & face_y & face_z)[0]

        # Re-index faces
        x_faces = faces[faces_to_keep, :].reshape(-1)
        idxs = np.searchsorted(vertices_to_keep, x_faces)
        new_faces = idxs.reshape(-1, 3)
    else:
        new_faces = faces

    # Create new gifti object
    normals = None
    normals_data = [da for da in gifti_surf.darrays
                    if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
    if normals_data:
        normals = normals_data[0].data[vertices_to_keep, :]
    new_gifti = _create_surf_gifti(new_vertices, new_faces, normals=normals)

    return new_gifti


def _find_non_manifold_edges(faces):
    """
    Detect non-manifold edges in a triangular mesh.

    A non-manifold edge is one that is shared by more than two faces, violating the
    manifold property of a well-formed surface mesh. This function identifies such
    edges and lists all faces that share them.

    Parameters
    ----------
    faces : np.ndarray, shape (F, 3)
        Array of triangular faces, where each row contains vertex indices defining a face.

    Returns
    -------
    non_manifold_edges : dict
        Dictionary mapping each non-manifold edge (as a sorted vertex index tuple)
        to the list of face indices that reference it.

    Notes
    -----
    - A manifold edge should be shared by exactly two faces.
    - This function is useful for mesh quality control and repair operations
      before further geometric processing.
    - Edge order is normalized by sorting vertex indices.

    Examples
    --------
    >>> faces = np.array([[0, 1, 2], [2, 1, 3], [0, 1, 3]])
    >>> _find_non_manifold_edges(faces)
    {(0, 1): [0, 2], (1, 2): [0, 1], (1, 3): [1, 2]}
    """

    edge_faces = defaultdict(list)

    for i, (vertex_1, vertex_2, vertex_3) in enumerate(faces):
        for edge in [(vertex_1, vertex_2), (vertex_2, vertex_3), (vertex_3, vertex_1)]:
            edge_faces[tuple(sorted(edge))].append(i)

    non_manifold_edges = {edge: fcs for edge, fcs in edge_faces.items() if len(fcs) > 2}
    return non_manifold_edges


def _fix_non_manifold_edges(vertices, faces):
    """
    Remove faces connected to non-manifold edges to ensure a topologically valid mesh.

    This function detects edges shared by more than two faces (non-manifold edges)
    and removes all faces that include such edges. The resulting mesh contains only
    manifold edges, improving stability for subsequent geometric processing steps.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Array of vertex coordinates.
    faces : np.ndarray, shape (F, 3)
        Array of triangular faces referencing vertex indices.

    Returns
    -------
    vertices : np.ndarray
        Original vertex array (unchanged).
    new_faces : np.ndarray
        Faces with all non-manifold-connected triangles removed.

    Notes
    -----
    - Non-manifold edges can disrupt mesh decimation, surface normal computation,
      and physical simulation workflows.
    - This function does not modify vertex geometry or reindex faces.
    - For further cleaning, `_remove_unconnected_vertices` can be used afterwards.

    Examples
    --------
    >>> v = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]])
    >>> f = np.array([[0,1,2], [1,2,3], [0,1,3]])  # (0,1) shared by 3 faces
    >>> _, new_f = _fix_non_manifold_edges(v, f)
    >>> new_f.shape[0]
    0
    """

    non_manifold_edges = _find_non_manifold_edges(faces)
    conflicting_faces = set()
    for faces_list in non_manifold_edges.values():
        conflicting_faces.update(faces_list)

    # Create a new face list excluding the conflicting faces
    new_faces = np.array(
        [face for i, face in enumerate(faces) if i not in conflicting_faces],
        dtype=np.int32
    )
    return vertices, new_faces


def _downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
    Downsample a GIFTI surface using VTK mesh decimation.

    This function reduces the number of vertices and faces in a GIFTI surface using
    VTK's `vtkDecimatePro` algorithm. The resulting surface preserves the overall
    geometry while simplifying mesh complexity according to the specified reduction
    factor.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        Input surface containing vertex (`NIFTI_INTENT_POINTSET`) and face
        (`NIFTI_INTENT_TRIANGLE`) data arrays.
    ds_factor : float, optional
        Fraction of vertices to retain. For example, `0.1` retains 10% of the
        original vertices. Default is 0.1.

    Returns
    -------
    new_gifti_surf : nibabel.gifti.GiftiImage
        Downsampled surface mesh with corresponding vertex and face arrays.

    Notes
    -----
    - Faces must be triangulated (three vertex indices per face).
    - If vertex normals are present, they are mapped from the original vertices
      to the nearest downsampled vertices.
    - The original `gifti_surf` is not modified; a new `GiftiImage` is returned.
    - This function requires a working VTK installation.

    Examples
    --------
    >>> gii = nib.load("lh.pial.converted.gii")
    >>> ds_gii = _downsample_single_surface(gii, ds_factor=0.2)
    >>> print(ds_gii.darrays[0].data.shape[0])
    20484
    """

    vertices = gifti_surf.darrays[0].data
    faces = gifti_surf.darrays[1].data

    # Convert vertices and faces to a VTK PolyData object
    points = vtkPoints()
    for point in vertices:
        points.InsertNextPoint(point)

    cells = vtkCellArray()
    for face in faces:
        cells.InsertNextCell(len(face))
        for vertex in face:
            cells.InsertCellPoint(vertex)

    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    # Apply vtkDecimatePro for decimation
    decimate = vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(1 - ds_factor)
    decimate.Update()

    # Extract the decimated mesh
    decimated_polydata = decimate.GetOutput()

    # Convert back to numpy arrays
    reduced_vertices = vtk_to_numpy(decimated_polydata.GetPoints().GetData())

    # Extract and reshape the face data
    face_data = vtk_to_numpy(decimated_polydata.GetPolys().GetData())
    # Assuming the mesh is triangulated, every fourth item is the size (3), followed by three
    # vertex indices
    reduced_faces = face_data.reshape(-1, 4)[:, 1:4]

    # Find the original vertices closest to the downsampled vertices
    kdtree = cKDTree(gifti_surf.darrays[0].data)
    _, orig_vert_idx = kdtree.query(reduced_vertices, k=1)
    orig_vert_idx = np.squeeze(orig_vert_idx)
    # enforce deterministic tie-break by sorting indices for identical distances
    orig_vert_idx = np.asarray(orig_vert_idx, dtype=int)

    reduced_normals = None
    if len(gifti_surf.darrays) > 2 and \
            gifti_surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
            gifti_surf.darrays[2].data.shape[0] == gifti_surf.darrays[0].data.shape[0]:
        reduced_normals = gifti_surf.darrays[2].data[orig_vert_idx]

    new_gifti_surf = _create_surf_gifti(reduced_vertices, reduced_faces, normals=reduced_normals)

    return new_gifti_surf


def _iterative_downsample_single_surface(gifti_surf, ds_factor=0.1):
    """
    Iteratively downsample a GIFTI surface mesh to a target vertex fraction.

    This function progressively reduces the number of vertices in a surface mesh
    using repeated applications of `_downsample_single_surface`, refining the
    downsampling ratio at each iteration until the target fraction of vertices is
    reached or closely approximated. Non-manifold edges and unconnected vertices
    are removed in post-processing.

    Parameters
    ----------
    gifti_surf : nibabel.gifti.GiftiImage
        Input surface containing vertex (`NIFTI_INTENT_POINTSET`) and face
        (`NIFTI_INTENT_TRIANGLE`) arrays.
    ds_factor : float, optional
        Target vertex retention fraction (e.g., `0.1` retains 10% of vertices).
        Default is 0.1.

    Returns
    -------
    current_surf : nibabel.gifti.GiftiImage
        The final downsampled surface, cleaned of non-manifold edges and
        unconnected vertices.

    Notes
    -----
    - The algorithm adaptively adjusts the decimation ratio per iteration to
      approach the target number of vertices without overshooting.
    - If the computed per-iteration reduction factor > 1, iteration stops to
      prevent upsampling.
    - Non-manifold and isolated vertices are automatically removed at the end.

    Examples
    --------
    >>> gii = nib.load("lh.pial.converted.gii")
    >>> ds_gii = _iterative_downsample_single_surface(gii, ds_factor=0.2)
    >>> print(ds_gii.darrays[0].data.shape)
    (20512, 3)
    """

    current_surf = gifti_surf
    current_vertices = gifti_surf.darrays[0].data.shape[0]
    target_vertices = int(current_vertices * ds_factor)
    current_ds_factor = target_vertices / current_vertices

    while current_vertices > target_vertices:
        # Downsample the mesh
        current_surf = _downsample_single_surface(current_surf, ds_factor=current_ds_factor)

        # Update the current vertices
        current_vertices = current_surf.darrays[0].data.shape[0]

        current_ds_factor = (target_vertices / current_vertices) * 1.25
        if current_ds_factor >= 1:
            break

    # Remove non-manifold edges
    ds_vertices = current_surf.darrays[0].data
    ds_faces = current_surf.darrays[1].data
    nonmani_vertices, nonmani_faces = _fix_non_manifold_edges(ds_vertices, ds_faces)

    normals = None
    if len(current_surf.darrays) > 2 and \
            current_surf.darrays[2].intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR'] and \
            current_surf.darrays[2].data.shape[0] == current_surf.darrays[0].data.shape[0]:
        normals = current_surf.darrays[2].data

    current_surf = _create_surf_gifti(nonmani_vertices, nonmani_faces, normals=normals)

    # Remove unconnected vertices
    current_surf = _remove_unconnected_vertices(current_surf)

    return current_surf


def _concatenate_surfaces(surfaces):
    """
    Concatenate multiple GIFTI surface meshes into a single mesh.

    This function merges several surface meshes by concatenating their vertex,
    face, and (if present) normal arrays. Face indices are re-indexed to match
    the combined vertex array, ensuring topological consistency across the merged
    surface.

    Parameters
    ----------
    surfaces : list of nibabel.gifti.GiftiImage
        List of GIFTI surfaces to combine, each containing vertex
        (`NIFTI_INTENT_POINTSET`) and face (`NIFTI_INTENT_TRIANGLE`) arrays.
        Normals (`NIFTI_INTENT_VECTOR`) are optional.

    Returns
    -------
    combined_surf : nibabel.gifti.GiftiImage
        Combined surface containing concatenated vertices, faces, and normals.

    Raises
    ------
    ValueError
        If any surface contains malformed vertex or face arrays.

    Notes
    -----
    - The order of concatenation determines the spatial layout of the resulting
      mesh (e.g., left- then right-hemisphere).
    - Normals are concatenated only if present in all input surfaces.
    - The input GIFTI objects are not modified in place.

    Examples
    --------
    >>> lh = nib.load("lh.pial.ds.gii")
    >>> rh = nib.load("rh.pial.ds.gii")
    >>> combined = _concatenate_surfaces([lh, rh])
    >>> print(combined.darrays[0].data.shape)
    (40962, 3)
    """

    combined_vertices = []
    combined_faces = []
    combined_normals = []

    face_offset = 0

    for mesh in surfaces:

        # Extract vertices and faces
        vertices = np.concatenate(
            [da.data for da in mesh.darrays
             if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']])
        faces = np.concatenate(
            [da.data for da in mesh.darrays
             if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']])

        # Check for normals
        normal_arrays = [da.data for da in mesh.darrays
                         if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_VECTOR']]
        normals = np.concatenate(normal_arrays) if normal_arrays else np.array([])

        combined_vertices.append(vertices)
        combined_faces.append(faces + face_offset)
        if normals.size:
            combined_normals.append(normals)

        face_offset += vertices.shape[0]

    # Combine the arrays
    combined_vertices = np.vstack(combined_vertices).astype(np.float32)
    combined_faces = np.vstack(combined_faces).astype(np.int32)
    if combined_normals:
        combined_normals = np.vstack(combined_normals).astype(np.float32)

    combined_surf = _create_surf_gifti(
        combined_vertices,
        combined_faces,
        normals=combined_normals
    )

    return combined_surf


__all__ = ["LayerSurfaceSet", "interpolate_data", "convert_fsaverage_to_native",
           "convert_native_to_fsaverage"]
