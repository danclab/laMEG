"""
Legacy laMEG surface converter
==============================

This script upgrades laminar surface directories from older laMEG versions to the current
standardized format compatible with the `LayerSurfaceSet` interface. It reconstructs the full
surface processing hierarchy?conversion, deep-vertex removal, hemisphere combination,
downsampling, and multilayer assembly?while generating complete metadata for each stage.

All outputs are written to `<SUBJECTS_DIR>/<subject_id>/surf/laminar`, ensuring consistency with
FreeSurfer directory structure and current laMEG conventions.

Typical usage:
    python convert_legacy_surfaces.py sub-104 /path/to/old_lameg_surf_dir

The resulting hierarchy can be directly validated and used via:
    >>> from lameg.surf import LayerSurfaceSet
    >>> surf_set = LayerSurfaceSet('sub-104', 11)
    >>> surf_set.validate()

Outputs
-------
For each layer and hemisphere, the following files are generated:

- **Raw conversion**
  - `lh.<layer>.raw.gii`, `rh.<layer>.raw.gii`

- **Coordinate-converted surfaces**
  - `lh.<layer>.converted.gii`, `rh.<layer>.converted.gii`
  - `<hemi>.<layer>.converted.json` (includes `t_orig`, `n_orig` matrices)

- **Deep-vertex?removed surfaces**
  - `lh.<layer>.nodeep.gii`, `rh.<layer>.nodeep.gii`
  - `<hemi>.<layer>.nodeep.json` (includes `deep_vertices_removed` indices)

- **Combined hemispheres**
  - `<layer>.combined.gii`
  - `<layer>.combined.json` (includes `lh_deep_vertices_removed`, `rh_deep_vertices_removed`)

- **Downsampled surfaces**
  - `<layer>.ds.gii`
  - `<layer>.ds.json` (includes `ds_factor`, `ds_removed_vertices`)

- **Orientation-specific surfaces**
  - `<layer>.ds.<orientation>.<fixed_flag>.gii`
  - `<layer>.ds.<orientation>.<fixed_flag>.json`
    (includes orientation method, fixed flag, and reference metadata)

- **Multilayer assemblies**
  - `multilayer.<n_layers>.ds.<orientation>.<fixed_flag>.gii`
  - `multilayer.<n_layers>.ds.<orientation>.<fixed_flag>.json`
    (aggregates per-layer metadata, orientation, and provenance)

Notes
-----
- Prefixes in legacy filenames (e.g. "sub-104_multilayer.11.ds.link_vector.fixed.gii")
  are stripped for consistency.
- Orientation methods supported: `link_vector`, `ds_surf_norm`, `orig_surf_norm`, `cps`.
- Metadata is automatically time-stamped and validated upon completion.
"""

import json
import shutil
import subprocess
from datetime import datetime

import os
import sys
import copy
import re
import numpy as np
import nibabel as nib

from lameg.surf import LayerSurfaceSet


# pylint: disable=R0912,R0915
from lameg.util import check_freesurfer_setup


def run(subj_id, old_lameg_surf_path, multilayer_n_layers=(2,11,15), fs_subjects_dir=None):
    """
    Convert and modernize laminar surface files from older laMEG directory structures.

    This utility reconstructs a complete laminar surface hierarchy for a subject, updating
    surface metadata and ensuring compatibility with the current `LayerSurfaceSet` class.
    It handles conversion to GIFTI format, deep vertex removal, hemisphere combination,
    downsampling, and orientation derivation, while inferring parameters for older naming
    conventions.

    The function replicates all standard laminar processing stages:
        - Raw conversion (`.raw.gii`)
        - Coordinate conversion (`.converted.gii`)
        - Deep-vertex removal (`.nodeep.gii`)
        - Hemisphere combination (`.combined.gii`)
        - Downsampling (`.ds.gii`)
        - Orientation computation (`.ds.<orientation>.<fixed_flag>.gii`)
        - Multilayer surface assembly (`multilayer.<n_layers>.ds.<orientation>.<fixed_flag>.gii`)

    Metadata files (`.json`) are generated or updated at each stage to include:
        - Coordinate transformation matrices (`t_orig`, `n_orig`)
        - Deep vertex indices removed per hemisphere
        - Downsampling ratio and removed vertices
        - Orientation method and fixed/free flag
        - Number of layers in multilayer reconstructions
        - Provenance of source files and modification timestamps

    Parameters
    ----------
    subj_id : str
        Subject identifier (must correspond to a FreeSurfer subject directory under
        `SUBJECTS_DIR`).
    old_lameg_surf_path : str
        Path to the directory containing older-format laminar surface files (pre-conversion).
    multilayer_n_layers : tuple of int, optional
        Number(s) of layers to reconstruct in multilayer surfaces. Defaults to (2, 11, 15).
    fs_subjects_dir : str or None, optional
        Path to the FreeSurfer subjects directory. If None, uses the environment variable
        `SUBJECTS_DIR`.

    Raises
    ------
    EnvironmentError
        If required FreeSurfer binaries (`mris_convert`, `mri_info`) are not available, or if
        `SUBJECTS_DIR` is undefined.
    FileNotFoundError
        If expected input surfaces or metadata files are missing.
    subprocess.CalledProcessError
        If FreeSurfer surface conversion commands fail.

    Notes
    -----
    - Old surface files may have arbitrary prefixes (e.g.
      "sub-104_multilayer.11.ds.link_vector.fixed.gii"). These prefixes are dropped when renaming
       to the standardized convention.
    - The function uses geometric comparison to infer which vertices were removed in deep-vertex
      cleaning and downsampling stages.
    - Orientation and fixed/free status are inferred from filenames or defaulted to
      `orientation='link_vector'` and `fixed=True` if not explicitly encoded.
    - The output hierarchy is validated at the end of execution using
      `LayerSurfaceSet.validate()`.

    Examples
    --------
    >>> run(
    ...     subj_id='sub-104',
    ...     old_lameg_surf_path='/data/old_surfaces/sub-104',
    ...     multilayer_n_layers=(2, 11),
    ... )
    """
    # --- Check that required FreeSurfer binaries are available ---
    check_freesurfer_setup()

    fs_subjects_dir = fs_subjects_dir or os.getenv('SUBJECTS_DIR')
    if fs_subjects_dir is None:
        raise EnvironmentError("SUBJECTS_DIR is not set and no subjects_dir was provided.")

    fs_subject_dir = os.path.join(fs_subjects_dir, subj_id)

    # Convert MRI to nii
    orig_mgz = os.path.join(fs_subject_dir, 'mri', 'orig.mgz')
    orig_nii = os.path.join(fs_subject_dir, 'mri', 'orig.nii')
    img = nib.load(orig_mgz)
    nib.save(img, orig_nii)

    fs_subject_surf_dir = os.path.join(fs_subject_dir, 'surf')

    # Create laminar surf dir
    out_dir = os.path.join(fs_subject_surf_dir, 'laminar')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    hemispheres = ['lh', 'rh']

    # Figure out layer names
    layer_names={}
    for n_layers in multilayer_n_layers:
        layer_names[n_layers]=[]
        layer_spacing = np.linspace(1, 0, n_layers)
        for layer in layer_spacing:
            if layer == 1:
                layer_names[n_layers].append('pial')
            elif layer == 0:
                layer_names[n_layers].append('white')
            else:
                layer_names[n_layers].append(f'{layer:.3f}')

    # Create out_dir/lh.0.714.raw.gii etc
    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        surfaces_to_process.append('inflated')
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                # Construct the original and new file names
                orig_name = os.path.join(fs_subject_surf_dir, f'{hemi}.{surface_name}')
                if os.path.exists(orig_name):
                    new_name = os.path.join(out_dir, f'{hemi}.{surface_name}.raw.gii')
                    if not os.path.exists(new_name):
                        # Convert the surface file to Gifti format
                        subprocess.run(['mris_convert', orig_name, new_name], check=True)

    # Copy lh.0.714.gii etc to out_dir/lh.0.714.converted.gii
    orig_mgz = os.path.join(fs_subject_dir, 'mri', 'orig.mgz')
    def _mat(flag):
        out = subprocess.check_output(['mri_info', flag, orig_mgz]).decode().split()
        return np.array([float(x) for x in out]).reshape(4, 4)
    t_orig = _mat('--vox2ras-tkr')  # vox -> tkRAS (mm)
    n_orig = _mat('--vox2ras')  # vox -> scanner RAS (mm)
    meta = {
        't_orig': t_orig.tolist(),
        'n_orig': n_orig.tolist()
    }
    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        surfaces_to_process.append('inflated')
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                orig_name = os.path.join(old_lameg_surf_path, f'{hemi}.{surface_name}.gii')
                if os.path.exists(orig_name):
                    new_name = os.path.join(out_dir, f'{hemi}.{surface_name}.converted.gii')
                    shutil.copy(orig_name, new_name)

                    # Create meta data
                    surf_meta = copy.deepcopy(meta)
                    meta_path = new_name.replace('.gii', '.json')
                    surf_meta['modified_at'] = datetime.now().isoformat()
                    with open(meta_path, 'w', encoding='utf-8') as file:
                        json.dump(surf_meta, file, indent=2)

    # Copy lh.0.714.nodeep.gii etc to out_dir/lh.0.714.nodeep.gii
    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        surfaces_to_process.append('inflated')
        for surface_name in surfaces_to_process:
            for hemi in hemispheres:
                orig_name = os.path.join(old_lameg_surf_path, f'{hemi}.{surface_name}.nodeep.gii')
                if os.path.exists(orig_name):
                    new_name = os.path.join(out_dir, f'{hemi}.{surface_name}.nodeep.gii')
                    shutil.copy(orig_name, new_name)

                    # Create meta data
                    surf_meta = copy.deepcopy(meta)

                    # Determine which vertices were removed
                    converted_surf = nib.load(
                        os.path.join(out_dir, f'{hemi}.{surface_name}.converted.gii')
                    )
                    nodeep_surf = nib.load(
                        os.path.join(out_dir, f'{hemi}.{surface_name}.nodeep.gii')
                    )
                    converted_vertices = converted_surf.darrays[0].data
                    nodeep_vertices = nodeep_surf.darrays[0].data

                    # Find the removed vertices by comparing coordinate sets
                    # Use rounding to mitigate floating-point drift between files
                    conv_rounded = np.round(converted_vertices, 6)
                    nodeep_rounded = np.round(nodeep_vertices, 6)

                    # Build a mask of retained vertices
                    nodeep_set = {tuple(v) for v in nodeep_rounded}

                    removed_verts = [i for i, v in enumerate(conv_rounded)
                                     if tuple(v) not in nodeep_set]

                    surf_meta['deep_vertices_removed'] = removed_verts
                    meta_path = new_name.replace('.gii', '.json')
                    surf_meta['modified_at'] = datetime.now().isoformat()
                    with open(meta_path, 'w', encoding='utf-8') as file:
                        json.dump(surf_meta, file, indent=2)

    # Copy 0.700.gii etc to out_dir/0.700.combined.gii
    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        surfaces_to_process.append('inflated')
        for surface_name in surfaces_to_process:
            orig_name = os.path.join(old_lameg_surf_path, f'{surface_name}.gii')
            if os.path.exists(orig_name):
                new_name = os.path.join(out_dir, f'{surface_name}.combined.gii')
                shutil.copy(orig_name, new_name)

                # Create meta data
                l_hemi_meta_path = os.path.join(out_dir, f'lh.{surface_name}.nodeep.json')
                with open(l_hemi_meta_path, 'r', encoding='utf-8') as file:
                    l_hemi_meta = json.load(file)
                r_hemi_meta_path = os.path.join(out_dir, f'rh.{surface_name}.nodeep.json')
                with open(r_hemi_meta_path, 'r', encoding='utf-8') as file:
                    r_hemi_meta = json.load(file)

                surf_meta = copy.copy(meta)
                surf_meta['lh_deep_vertices_removed'] = l_hemi_meta['deep_vertices_removed']
                surf_meta['rh_deep_vertices_removed'] = r_hemi_meta['deep_vertices_removed']

                meta_path = new_name.replace('.gii', '.json')
                surf_meta['modified_at'] = datetime.now().isoformat()
                with open(meta_path, 'w', encoding='utf-8') as file:
                    json.dump(surf_meta, file, indent=2)

    # Copy 0.700.ds.gii etc to out_dir/0.700.ds.gii
    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        surfaces_to_process.append('inflated')
        for surface_name in surfaces_to_process:
            orig_name = os.path.join(old_lameg_surf_path, f'{surface_name}.ds.gii')
            if os.path.exists(orig_name):
                new_name = os.path.join(out_dir, f'{surface_name}.ds.gii')
                shutil.copy(orig_name, new_name)

                # Create meta data
                # Determine which vertices were removed
                combined_surf = nib.load(os.path.join(out_dir, f'{surface_name}.combined.gii'))
                ds_surf = nib.load(os.path.join(out_dir, f'{surface_name}.ds.gii'))
                combined_vertices = combined_surf.darrays[0].data
                ds_vertices = ds_surf.darrays[0].data

                # Find the removed vertices by comparing coordinate sets
                # Use rounding to mitigate floating-point drift between files
                combined_rounded = np.round(combined_vertices, 6)
                ds_rounded = np.round(ds_vertices, 6)

                # Build a mask of retained vertices
                ds_set = {tuple(v) for v in ds_rounded}

                removed_verts = [i for i, v in enumerate(combined_rounded)
                                 if tuple(v) not in ds_set]
                ds_factor = ds_vertices.shape[0]/combined_vertices.shape[0]
                meta['ds_factor'] = ds_factor

                combined_meta_path = os.path.join(out_dir, f'{surface_name}.combined.json')
                with open(combined_meta_path, 'r', encoding='utf-8') as file:
                    combined_meta = json.load(file)

                surf_meta = copy.deepcopy(combined_meta)
                surf_meta['ds_factor'] = ds_factor
                surf_meta['ds_removed_vertices'] = removed_verts
                meta_path = new_name.replace('.gii', '.json')
                surf_meta['modified_at'] = datetime.now().isoformat()
                with open(meta_path, 'w', encoding='utf-8') as file:
                    json.dump(surf_meta, file, indent=2)

    # Figure out which orientation settings were used
    # Copy 0.700.ds.linked_vector.fixed.gii etc to out_dir/0.700.ds.linked_vector.fixed.gii
    orientation_suffixes = ['link_vector', 'ds_surf_norm', 'orig_surf_norm', 'cps']

    for n_layers in multilayer_n_layers:
        surfaces_to_process = copy.copy(layer_names[n_layers])
        for surface_name in surfaces_to_process:
            # Search for any matching orientation files in the old directory
            for orientation in orientation_suffixes:
                for fixed_flag in ['fixed', 'not_fixed']:
                    fname = f'{surface_name}.ds.{orientation}.{fixed_flag}.gii'
                    src_path = os.path.join(old_lameg_surf_path, fname)
                    if os.path.exists(src_path):
                        dest_path = os.path.join(out_dir, fname)
                        shutil.copy(src_path, dest_path)

                        # Create meta data
                        # Load metadata from the corresponding downsampled JSON
                        ds_meta_path = os.path.join(out_dir, f'{surface_name}.ds.json')
                        with open(ds_meta_path, 'r', encoding='utf-8') as file:
                            ds_meta = json.load(file)

                        # Create updated metadata
                        surf_meta = copy.deepcopy(ds_meta)
                        surf_meta['orientation'] = orientation
                        surf_meta['fix_orientation'] = fixed_flag == 'fixed'
                        surf_meta['modified_at'] = datetime.now().isoformat()

                        meta_path = dest_path.replace('.gii', '.json')
                        with open(meta_path, 'w', encoding='utf-8') as file:
                            json.dump(surf_meta, file, indent=2)


    # Copy multilayer.11.ds.link_vector.fixed.gii etc to
    # out_dir/multilayer.11.ds.link_vector.fixed.gii
    all_gii_files = [f for f in os.listdir(old_lameg_surf_path) if f.endswith('.gii')]
    multilayer_files = [f for f in all_gii_files
                        if 'multilayer' in f and '.ds' in f and 'warped' not in f]

    multilayer_pattern = re.compile(
        r'.*multilayer\.?(\d+)?\.ds(?:\.([A-Za-z_]+))?(?:\.(fixed|not_fixed))?\.gii'
    )

    for fname in multilayer_files:
        original_fname = fname
        # Drop any prefix before "multilayer"
        stripped_fname = fname[fname.find('multilayer'):] if 'multilayer' in fname else fname

        match = multilayer_pattern.match(stripped_fname)
        if match:
            orientation = match.group(2) \
                if match.group(2) in orientation_suffixes else 'link_vector'
            fixed_flag = match.group(3) \
                if match.group(3) in ['fixed', 'not_fixed'] else 'fixed'

            pial_ds = nib.load(os.path.join(out_dir, 'pial.ds.gii'))
            pial_vertices = pial_ds.darrays[0].data
            multilayer_ds = nib.load(os.path.join(old_lameg_surf_path, original_fname))
            multilayer_vertices = multilayer_ds.darrays[0].data
            n_layers = int(multilayer_vertices.shape[0]/pial_vertices.shape[0])

            # Build standardized output filename
            standardized_name = f"multilayer.{n_layers}.ds.{orientation}.{fixed_flag}.gii"

            src_path = os.path.join(old_lameg_surf_path, original_fname)
            dest_path = os.path.join(out_dir, standardized_name)
            shutil.copy(src_path, dest_path)

            # Create meta data
            multilayer_meta = copy.deepcopy(meta)
            multilayer_meta['n_layers'] = n_layers
            multilayer_meta['orientation'] = orientation
            multilayer_meta['fix_orientation'] = fixed_flag == 'fixed'
            for layer_name in layer_names[n_layers]:
                surf_meta_path = os.path.join(
                    out_dir,
                    f'{layer_name}.ds.{orientation}.{fixed_flag}.json'
                )
                with open(surf_meta_path, 'r', encoding='utf-8') as file:
                    surf_meta = json.load(file)
                multilayer_meta[f'lh_{layer_name}_deep_vertices_removed'] = \
                    surf_meta['lh_deep_vertices_removed']
                multilayer_meta[f'rh_{layer_name}_deep_vertices_removed'] = \
                    surf_meta['rh_deep_vertices_removed']
                multilayer_meta[f'{layer_name}_ds_removed_vertices'] = \
                    surf_meta['ds_removed_vertices']
            multilayer_meta['source_file'] = original_fname
            multilayer_meta['modified_at'] = datetime.now().isoformat()

            meta_path = dest_path.replace('.gii', '.json')
            with open(meta_path, 'w', encoding='utf-8') as file:
                json.dump(multilayer_meta, file, indent=2)

            surf_set = LayerSurfaceSet(subj_id, n_layers)
            surf_set.validate(['raw', 'converted', 'nodeep', 'combined', 'ds'], hemis=('lh', 'rh'),
                              orientations=[orientation], fixed=fixed_flag=='fixed')

    all_fwhm_files = [f for f in os.listdir(old_lameg_surf_path) if f.startswith('FWHM')]
    for fwhm_file in all_fwhm_files:
        src = os.path.join(old_lameg_surf_path, fwhm_file)
        dest = os.path.join(out_dir, fwhm_file)
        shutil.copy(src, dest)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert legacy laMEG surface directories to standardized multilayer format."
    )
    parser.add_argument("subject_id", help="FreeSurfer subject ID (e.g. sub-104)")
    parser.add_argument("old_lameg_surf_path", help="Path to legacy laMEG surface directory")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[2, 11, 15],
        help="List of multilayer n_layers values to reconstruct (default: 2 11 15)"
    )
    parser.add_argument(
        "--fsdir",
        type=str,
        default=None,
        help="Path to FreeSurfer SUBJECTS_DIR (optional; defaults to $SUBJECTS_DIR)"
    )

    args = parser.parse_args()

    subject = args.subject_id
    old_path = os.path.abspath(args.old_lameg_surf_path)
    n_layers_tup = tuple(args.layers)
    fsdir = args.fsdir

    print(f"Converting legacy surfaces for {subject}...")
    print(f"  Legacy directory: {old_path}")
    print(f"  Layers to reconstruct: {n_layers_tup}")
    if fsdir:
        print(f"  FreeSurfer SUBJECTS_DIR: {fsdir}")

    try:
        run(subject, old_path, multilayer_n_layers=n_layers_tup, fs_subjects_dir=fsdir)

        output_dir = os.path.join(
            os.path.join(fsdir or os.getenv('SUBJECTS_DIR'), subject),
            'surf',
            'laminar'
        )

        print(f"\nSurfaces successfully converted and copied to:\n  {output_dir}")
        print(f"\nYou may now verify the results and, if satisfied, "
              f"delete the old directory:\n  {old_path}")

    # pylint: disable=W0718
    except Exception as e:
        print(f"\n Conversion failed: {e}")
        sys.exit(1)
