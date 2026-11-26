"""Fitted tetrahedral DUNEuro MEG forward calculation on a SimNIBS CHARM mesh."""

from dataclasses import dataclass

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
# VTK exposes these classes through compiled extension modules.
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints  # pylint: disable=no-name-in-module
from vtkmodules.vtkCommonDataModel import (  # pylint: disable=no-name-in-module
    VTK_TETRA, vtkCellArray, vtkStaticCellLocator, vtkUnstructuredGrid
)


# Isotropic tissue conductivities in S/m from the SimNIBS standard conductivity
# table (simnibs.github.io/simnibs/build/html/documentation/conductivity.html).
# Keys are CHARM volume tags; surface tags are not conductor compartments.
_CHARM_CONDUCTIVITY = {
    1: 0.126, 2: 0.275, 3: 1.654, 4: 0.01, 5: 0.465,  # WM, GM, CSF, bone, scalp
    6: 0.5, 7: 0.008, 8: 0.025, 9: 0.6, 10: 0.16,  # eyes, compact/spongy bone, blood, muscle
}


@dataclass
class _CharmMesh:
    """CHARM's tetrahedra retain their anatomical tags; coordinates are in metres."""

    points_m: np.ndarray
    tetrahedra: np.ndarray
    tissue_tags: np.ndarray
    conductivities: dict


def read_charm_mesh(mesh_fname, conductivities=None):
    """Read only volume elements and their physical tissue tags from a Gmsh mesh."""
    try:
        import meshio  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("Reading a CHARM FEM mesh requires the optional package meshio.") from exc

    gmsh = meshio.read(str(mesh_fname))
    # Surface triangles carry different labels (e.g. 1002 for a GM boundary).
    # Only the tetrahedra enter the volume conductor and the conductivity table.
    if any(c.dim == 3 and c.type != 'tetra' for c in gmsh.cells):
        raise ValueError("All volume elements in the CHARM mesh must be linear tetrahedra.")
    tetrahedra = gmsh.get_cells_type('tetra')
    if tetrahedra.size == 0:
        raise ValueError("The CHARM mesh must contain linear tetrahedra.")
    try:
        tags = gmsh.get_cell_data('gmsh:physical', 'tetra')
    except KeyError as exc:
        raise ValueError("The tetrahedra must have Gmsh physical tissue tags.") from exc
    if tags.shape != (len(tetrahedra),) or tags.dtype.kind not in 'iu' or np.any(tags <= 0):
        raise ValueError("Expected one positive CHARM physical tissue tag per tetrahedron.")
    present = np.unique(tags)
    if conductivities is None:
        conductivities = _CHARM_CONDUCTIVITY
    missing = set(present).difference(conductivities)
    if missing:
        raise ValueError(f"Missing conductivities (S/m) for CHARM tissue tags {sorted(missing)}.")
    table = {int(tag): float(conductivities[tag]) for tag in present}
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in table.values()):
        raise ValueError("Tissue conductivities must be finite and strictly positive (S/m).")

    points = np.asarray(gmsh.points, dtype=float) / 1000  # native T1 world mm -> m
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or tetrahedra.min() < 0 or tetrahedra.max() >= len(points)):
        raise ValueError("Invalid coordinates or node indices in the tetrahedral mesh.")
    # Some Gmsh files also contain nodes used only by lower-dimensional objects.
    # Passing these through would leave FEM degrees of freedom without elements.
    used = np.zeros(len(points), dtype=bool)
    used[tetrahedra] = True
    if not used.all():
        tetrahedra = (np.cumsum(used) - 1)[tetrahedra]
        points = points[used]
    return _CharmMesh(points, tetrahedra, tags, table)


def source_compartments(mesh, source_rr_m):
    """Locate original sources in tetrahedra; zero means outside or across a tissue interface."""
    # VTK's spatial index searches the actual CHARM cells. Re-triangulating the
    # nodes or looking up the nearest centroid would change the tissue boundaries.
    points = vtkPoints()
    points.SetData(numpy_to_vtk(mesh.points_m))
    connectivity = np.ascontiguousarray(mesh.tetrahedra, dtype=np.int64)
    cells = vtkCellArray()
    offsets = 4 * np.arange(len(connectivity) + 1, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets),
                  numpy_to_vtkIdTypeArray(connectivity.ravel()))
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.SetCells(VTK_TETRA, cells)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(grid)
    locator.BuildLocator()
    indices = np.fromiter((locator.FindCell(p) for p in source_rr_m), dtype=np.int64,
                          count=len(source_rr_m))
    tags = np.zeros(len(indices), dtype=mesh.tissue_tags.dtype)
    inside = np.flatnonzero(indices >= 0)
    tags[inside] = mesh.tissue_tags[indices[inside]]

    # Exactly on a shared face, edge or node, VTK and DUNEuro may legitimately
    # return different containing elements. Accept a unique tissue, never a
    # fortuitous GM label. The barycentric tolerance is dimensionless.
    nodes, neighbors = vtkIdList(), vtkIdList()
    for start in range(0, len(inside), 8192):
        selected = inside[start:start + 8192]
        corners = mesh.tetrahedra[indices[selected]]
        xyz = mesh.points_m[corners]
        basis = (xyz[:, 1:] - xyz[:, :1]).transpose(0, 2, 1)
        delta = (source_rr_m[selected] - xyz[:, 0])[..., None]
        local = np.linalg.solve(basis, delta)[..., 0]
        weights = np.column_stack((1 - local.sum(axis=1), local))
        interior = weights > 1e-9
        # FindCell may accept a point infinitesimally beyond an element.
        contained = np.all(weights >= -1e-9, axis=1)
        tags[selected[~contained]] = 0
        for row in np.flatnonzero(contained & ~interior.all(axis=1)):
            nodes.Reset()
            for node in corners[row, interior[row]]:
                nodes.InsertNextId(int(node))
            index = selected[row]
            grid.GetCellNeighbors(indices[index], nodes, neighbors)
            if any(mesh.tissue_tags[neighbors.GetId(i)] != tags[index]
                   for i in range(neighbors.GetNumberOfIds())):
                tags[index] = 0
    return tags


def _bindings():
    """Load the optional Python extension built alongside the DUNEuro C++ solver."""
    try:
        import duneuropy  # pylint: disable=import-outside-toplevel,import-error
    except ImportError as exc:
        raise ImportError("Build the duneuro-py bindings and add the directory containing "
                          "the duneuropy extension to PYTHONPATH.") from exc
    return duneuropy


def _driver_config(mesh):
    # The in-memory interface uses zero-based node indices and dense material
    # indices into the conductivity vector, not the original CHARM tag numbers.
    tissue_ids = np.array(sorted(mesh.conductivities))
    return {
        'type': 'fitted', 'solver_type': 'cg', 'element_type': 'tetrahedron',
        # Upstream explicitly gates the patch-based Venant source model.
        'enable_experimental': True,
        'volume_conductor': {
            'grid': {'nodes': mesh.points_m, 'elements': mesh.tetrahedra},
            'tensors': {'labels': np.searchsorted(tissue_ids, mesh.tissue_tags),
                        'conductivities': np.array([mesh.conductivities[t] for t in tissue_ids])}
        },
        'solver': {'verbose': 0},
        # Older DUNEuro builds cache a mesh-sized flux integral for every
        # pickup when the coils are registered. The transfer solve needs each
        # integral only once, so assemble it on demand on the full mesh.
        'meg': {'type': 'physical', 'intorderadd': 3, 'cache': {'enable': False}}
    }


def _primary_field(positions, moments, coils, projections):
    """Analytic dipole field along each pickup axis, without the common mu0/(4 pi) factor."""
    delta = coils[None, :, :] - positions[:, None, :]
    radii = np.linalg.norm(delta, axis=2)
    if np.any(radii == 0):
        raise ValueError("A dipole and pickup coil coincide.")
    return np.einsum('scd,cd->sc', np.cross(moments[:, None, :], delta),
                     projections) / radii**3


def solve_forward(mesh, source_rr_m, source_nn, coil_rr_m, coil_nn, channel_weights):
    """Return channels x signed sources in physical SI units per A m using one transfer matrix."""
    dune = _bindings()
    driver = dune.MEEGDriver3d(_driver_config(mesh))
    # Each physical pickup has one projection direction. The signed channel
    # montage (including compensation) is applied after adding its primary field.
    coils = [dune.FieldVector3D(p.tolist()) for p in coil_rr_m]
    projections = [[dune.FieldVector3D(n.tolist())] for n in coil_nn]
    driver.setCoilsAndProjections(coils, projections)
    # Each parallel transfer solve allocates mesh-sized work arrays. Older
    # bindings read the worker count from solver; newer ones read it here
    # at the top level. Set both when solving on a full CHARM head.
    transfer, _ = driver.computeMEGTransferMatrix(
        {'solver': {'reduction': 1e-10, 'numberOfThreads': 1}, 'numberOfThreads': 1})

    # Match the 20-mm St Venant reference length to our mesh's SI coordinates.
    # The patch is assembled from the closest node's neighboring tetrahedra of
    # the same conductivity as the containing cell, without moving the dipole.
    source_model = {'type': 'patch_based_venant', 'initialization': 'closest_vertex',
                    'restrict': True, 'numberOfMoments': 3, 'mixedMoments': True,
                    'referenceLength': 0.020, 'relaxationFactor': 1e-6,
                    'weightingExponent': 1, 'quadratureRuleOrder': 2}
    # A St. Venant worker also needs a mesh-wide element-neighborhood map.
    config = {'source_model': source_model, 'post_process_meg': False, 'numberOfThreads': 1}
    gain = np.empty((channel_weights.shape[0], len(source_rr_m)))
    for start in range(0, len(source_rr_m), 256):
        block = slice(start, start + 256)
        positions, moments = source_rr_m[block], source_nn[block]
        dipoles = [dune.Dipole3d(p.tolist(), n.tolist()) for p, n in zip(positions, moments)]
        secondary, _ = driver.applyMEGTransfer(transfer, dipoles, config)
        secondary = np.asarray(secondary)
        if secondary.shape != (len(dipoles), len(coils)) or not np.isfinite(secondary).all():
            raise ValueError("DUNEuro returned an invalid dipole-by-coil field.")
        # DUNEuro integrates +sigma*grad(u), whereas the physical volume
        # current is -sigma*grad(u). With distances in m and moments in A m,
        # mu0/(4 pi) converts their difference into tesla at each pickup.
        primary = _primary_field(positions, moments, coil_rr_m, coil_nn)
        gain[:, block] = channel_weights @ ((primary - secondary).T * 1e-7)
    return gain
