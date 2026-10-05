"""Validate BigBrain-informed layer-to-lamina mapping under boundary uncertainty.

The practical laMEG case uses 11 equidistant reconstructed surfaces and maps them to
BigBrain-defined laminae. This validation perturbs the six positive laminar thicknesses in log
space, renormalizes them to one cortical thickness, and calibrates the perturbation amplitude to a
requested global RMS displacement of the five internal laminar boundaries.

Three error sources are evaluated: finite-surface sampling with the true perturbed boundaries,
boundary mismatch with exact continuous profiles, and their combined practical effect.
"""

# This standalone scientific validation script is intentionally self-contained.
# pylint: disable=too-many-lines
# pylint: disable=duplicate-code

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import nnls
from scipy.special import erf  # pylint: disable=no-name-in-module

from lameg.laminar import compute_bigbrain_laminar_weights, surface_to_laminae
from lameg.surf import LayerSurfaceSet

LAMINA_LABELS = ("I", "II", "III", "IV", "V", "VI")
BOUNDARY_LABELS = ("I/II", "II/III", "III/IV", "IV/V", "V/VI")
DEFAULT_SIGMAS = (0.025, 0.05, 0.10, 0.20)
DEFAULT_BOUNDARY_RMS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10)
DEFAULT_BOUNDARY_WEIGHTS = (0.7, 1.0, 1.4, 1.4, 1.3)


class CachedBoundarySurfaceSet:  # pylint: disable=too-few-public-methods
    """Minimal surface-set interface backed by cached laminar boundaries."""

    def __init__(self, n_layers, edges):
        """Store reconstructed layer spacing and cached cumulative boundaries."""
        self.layer_spacing = np.linspace(1.0, 0.0, int(n_layers))
        self._boundaries = np.asarray(edges[:, 1:], dtype=float).T

    def get_bigbrain_layer_boundaries(self, subj_coord=None):
        """Return cached BigBrain boundaries."""
        if subj_coord is not None:
            raise ValueError("Cached boundaries do not accept `subj_coord`.")
        return self._boundaries


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate BigBrain-informed layer-to-lamina mapping uncertainty."
    )
    parser.add_argument("--subject", default="sub-104")
    parser.add_argument("--subjects-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-columns", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-centers", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument(
        "--boundary-rms",
        type=float,
        nargs="+",
        default=list(DEFAULT_BOUNDARY_RMS),
        help="Target global RMS internal-boundary displacement as cortical-thickness fraction.",
    )
    parser.add_argument(
        "--boundary-weights",
        type=float,
        nargs=5,
        default=list(DEFAULT_BOUNDARY_WEIGHTS),
        metavar=("I_II", "II_III", "III_IV", "IV_V", "V_VI"),
        help="Relative RMS uncertainty targets for the five internal laminar boundaries.",
    )
    return parser.parse_args()


def repo_root():
    """Return the repository root inferred from this script location."""
    return Path(__file__).resolve().parents[2]


def resolve_subjects_dir(value):
    """Resolve the FreeSurfer SUBJECTS_DIR."""
    if value is not None:
        return Path(value).expanduser().resolve()
    if os.getenv("SUBJECTS_DIR"):
        return Path(os.environ["SUBJECTS_DIR"]).expanduser().resolve()
    return (repo_root() / "test_data" / "fs").resolve()


def resolve_output_dir(value):
    """Resolve the validation output directory."""
    if value is not None:
        return Path(value).expanduser().resolve()
    path = repo_root() / "validation" / "bigbrain_mapping" / "layer_to_lamina_mapping_output"
    return path.resolve()


def validate_args(args):
    """Validate command-line arguments."""
    if args.n_columns < 0:
        raise ValueError("--n-columns must be >= 0.")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1.")
    if args.n_centers < 2:
        raise ValueError("--n-centers must be >= 2.")

    sigmas = np.asarray(args.sigmas, dtype=float)
    uncertainties = np.asarray(args.boundary_rms, dtype=float)
    weights = np.asarray(args.boundary_weights, dtype=float)
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("All sigma values must be finite and > 0.")
    if np.any(~np.isfinite(uncertainties)) or np.any(uncertainties < 0):
        raise ValueError("All boundary RMS values must be finite and >= 0.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("All boundary weights must be finite and > 0.")


def choose_candidate_columns(surf_set, requested, seed):
    """Choose cortical columns, oversampling slightly to allow invalid atlas mappings."""
    pial = surf_set.load("pial", stage="ds")
    n_total = int(pial.darrays[0].data.shape[0])
    if requested == 0 or requested >= n_total:
        return np.arange(n_total, dtype=int), n_total

    n_candidates = min(n_total, max(requested, int(np.ceil(requested * 1.25))))
    rng = np.random.default_rng(seed)
    columns = np.sort(rng.choice(n_total, size=n_candidates, replace=False))
    return columns.astype(int), n_total


def get_atlas_edges(surf_set, candidate_columns, requested):
    """Map BigBrain boundaries and retain valid, strictly increasing cortical columns."""
    edges, _ = compute_bigbrain_laminar_weights(surf_set, columns=candidate_columns)
    edges = np.asarray(edges, dtype=float)
    valid = np.all(np.isfinite(edges), axis=1)
    valid &= np.all(np.diff(edges, axis=1) > 0, axis=1)
    columns, edges = candidate_columns[valid], edges[valid]

    if requested > 0:
        columns, edges = columns[:requested], edges[:requested]
    if len(columns) == 0:
        raise RuntimeError("No valid BigBrain cortical columns were found.")
    return columns, edges


def weights_for_edges(edges, n_layers=11):
    """Compute laMEG layer-to-lamina weights for supplied laminar edges."""
    proxy = CachedBoundarySurfaceSet(n_layers, edges)
    returned_edges, weights = compute_bigbrain_laminar_weights(proxy)
    np.testing.assert_allclose(returned_edges, edges, rtol=0.0, atol=1e-12)
    source_depth = 1.0 - np.asarray(proxy.layer_spacing, dtype=float)
    return source_depth, np.asarray(weights, dtype=float)


def sample_gaussian(source_depth, centers, sigma):
    """Sample Gaussian cortical-depth profiles at reconstructed surface depths."""
    return np.exp(-0.5 * ((source_depth[:, None] - centers[None, :]) / sigma) ** 2)


def exact_gaussian_laminar_mean(edges, centers, sigma):
    """Return exact continuous Gaussian means within each lamina."""
    lower = edges[:, :-1, None]
    upper = edges[:, 1:, None]
    center = centers[None, None, :]
    scale = np.sqrt(2.0) * sigma
    integral = sigma * np.sqrt(np.pi / 2.0) * (
        erf((upper - center) / scale) - erf((lower - center) / scale)
    )
    return integral / (upper - lower)


def reconstruct_laminar_profile(sampled_profile, weights):
    """Map sampled layer profiles to laminae for every cortical column."""
    n_columns = weights.shape[0]
    n_layers, n_centers = sampled_profile.shape
    layer_data = np.broadcast_to(sampled_profile[:, None, :], (n_layers, n_columns, n_centers))
    estimated = surface_to_laminae(layer_data, weights)
    return np.transpose(estimated, (1, 0, 2))


def profile_metrics(estimated, truth):
    """Compute normalized profile error and dominant-lamina accuracy."""
    error = estimated - truth
    truth_scale = np.maximum(np.max(np.abs(truth), axis=1), np.finfo(float).eps)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    nrmse = rmse / truth_scale
    true_dominant = np.argmax(truth, axis=1)
    estimated_dominant = np.argmax(estimated, axis=1)
    return {
        "median_nrmse": float(np.median(nrmse)),
        "mean_nrmse": float(np.mean(nrmse)),
        "dominant_accuracy": float(np.mean(true_dominant == estimated_dominant)),
    }


def dominant_confusion(estimated, truth):
    """Count true versus estimated dominant laminae."""
    true_dominant = np.argmax(truth, axis=1).ravel()
    estimated_dominant = np.argmax(estimated, axis=1).ravel()
    n_laminae = len(LAMINA_LABELS)
    encoded = true_dominant * n_laminae + estimated_dominant
    return np.bincount(encoded, minlength=n_laminae**2).reshape(n_laminae, n_laminae)


def build_boundary_jacobian(atlas_edges):
    """Build the linearized log-thickness-to-boundary displacement Jacobian."""
    thickness = np.diff(atlas_edges, axis=1)
    boundaries = atlas_edges[:, 1:-1]
    jacobian = np.empty((atlas_edges.shape[0], 5, 6), dtype=float)
    for boundary_idx in range(5):
        for lamina_idx in range(6):
            indicator = lamina_idx <= boundary_idx
            jacobian[:, boundary_idx, lamina_idx] = thickness[:, lamina_idx] * (
                indicator - boundaries[:, boundary_idx]
            )
    return jacobian


def calibrate_log_thickness_weights(atlas_edges, target_boundary_weights):
    """Calibrate six log-thickness SD weights to five relative boundary RMS targets."""
    jacobian = build_boundary_jacobian(atlas_edges)
    calibration_matrix = np.mean(jacobian**2, axis=0)
    target_profile = np.array(target_boundary_weights, dtype=float, copy=True)
    target_profile /= np.sqrt(np.mean(target_profile**2))
    target_variance = target_profile**2

    initial_variance = np.linalg.pinv(calibration_matrix) @ target_variance
    null_basis = null_space(calibration_matrix)
    projection = np.eye(6) - np.ones((6, 6)) / 6.0
    if null_basis.shape[1] > 0:
        null_coefficients, *_ = np.linalg.lstsq(
            projection @ null_basis,
            -(projection @ initial_variance),
            rcond=None,
        )
        layer_variance = initial_variance + null_basis @ null_coefficients
    else:
        layer_variance = initial_variance

    residual = np.linalg.norm(calibration_matrix @ layer_variance - target_variance)
    if np.min(layer_variance) < -1e-8 or residual > 1e-8:
        layer_variance, _ = nnls(calibration_matrix, target_variance)
        method = "non-negative least-squares fallback"
    else:
        layer_variance = np.maximum(layer_variance, 0.0)
        method = "exact solution closest to uniform layer variance"

    if not np.any(layer_variance > 0):
        raise RuntimeError("Could not calibrate log-thickness perturbation variances.")

    log_sd_weights = np.sqrt(layer_variance)
    log_sd_weights /= np.sqrt(np.mean(log_sd_weights**2))
    predicted_variance = calibration_matrix @ (log_sd_weights**2)
    predicted_profile = np.sqrt(predicted_variance)
    predicted_profile /= np.sqrt(np.mean(predicted_profile**2))
    return log_sd_weights, target_profile, predicted_profile, calibration_matrix, method


def edges_from_log_thickness_perturbation(atlas_edges, random_field, scale):
    """Perturb positive laminar thicknesses in log space while preserving total thickness."""
    thickness = np.diff(atlas_edges, axis=1)
    perturbed_log = np.log(thickness) + scale * random_field
    perturbed_log -= np.max(perturbed_log, axis=1, keepdims=True)
    perturbed_thickness = np.exp(perturbed_log)
    perturbed_thickness /= np.sum(perturbed_thickness, axis=1, keepdims=True)
    edges = np.zeros_like(atlas_edges)
    edges[:, 1:] = np.cumsum(perturbed_thickness, axis=1)
    edges[:, -1] = 1.0
    return edges


def normalized_boundary_rms(atlas_edges, perturbed_edges):
    """Return global RMS displacement of the five internal boundaries."""
    displacement = perturbed_edges[:, 1:-1] - atlas_edges[:, 1:-1]
    return float(np.sqrt(np.mean(displacement**2)))


def perturb_edges_to_target_rms(
    atlas_edges,
    target_rms,
    log_sd_weights,
    rng,
    tolerance=1e-6,
    max_iterations=80,
):
    """Generate one log-thickness perturbation realization matching a target global RMS."""
    if target_rms == 0:
        return atlas_edges.copy(), 0.0, 0.0

    random_field = rng.standard_normal((atlas_edges.shape[0], 6))
    random_field *= log_sd_weights[None, :]
    lower_scale, upper_scale = 0.0, 0.25
    perturbed = edges_from_log_thickness_perturbation(atlas_edges, random_field, upper_scale)
    achieved = normalized_boundary_rms(atlas_edges, perturbed)

    while achieved < target_rms and upper_scale < 128.0:
        upper_scale *= 2.0
        perturbed = edges_from_log_thickness_perturbation(
            atlas_edges, random_field, upper_scale
        )
        achieved = normalized_boundary_rms(atlas_edges, perturbed)
    if achieved < target_rms:
        raise RuntimeError(
            f"Could not reach target boundary RMS {target_rms:.4f}; maximum was {achieved:.4f}."
        )

    best_edges, best_rms, best_scale = perturbed, achieved, upper_scale
    for _ in range(max_iterations):
        scale = (lower_scale + upper_scale) / 2.0
        perturbed = edges_from_log_thickness_perturbation(atlas_edges, random_field, scale)
        achieved = normalized_boundary_rms(atlas_edges, perturbed)
        best_edges, best_rms, best_scale = perturbed, achieved, scale
        if abs(achieved - target_rms) < tolerance:
            break
        if achieved < target_rms:
            lower_scale = scale
        else:
            upper_scale = scale
    return best_edges, best_rms, best_scale


def physical_boundary_displacement(atlas_edges, true_edges, cortical_thickness_mm):
    """Summarize boundary displacement after conversion to micrometres."""
    displacement_fraction = true_edges[:, 1:-1] - atlas_edges[:, 1:-1]
    displacement_um = displacement_fraction * cortical_thickness_mm[:, None] * 1000.0
    absolute_um = np.abs(displacement_um)
    return {
        "rms_um": float(np.sqrt(np.mean(displacement_um**2))),
        "median_abs_um": float(np.median(absolute_um)),
        "q95_abs_um": float(np.quantile(absolute_um, 0.95)),
        "max_abs_um": float(np.max(absolute_um)),
    }


def boundary_specific_rms(atlas_edges, true_edges, cortical_thickness_mm):
    """Return boundary-specific RMS displacement in normalized depth and micrometres."""
    displacement_fraction = true_edges[:, 1:-1] - atlas_edges[:, 1:-1]
    displacement_um = displacement_fraction * cortical_thickness_mm[:, None] * 1000.0
    rms_fraction = np.sqrt(np.mean(displacement_fraction**2, axis=0))
    rms_um = np.sqrt(np.mean(displacement_um**2, axis=0))
    return rms_fraction, rms_um


def lamina_thickness_statistics(edges, cortical_thickness_mm):
    """Return robust lower-tail and median laminar-thickness diagnostics."""
    thickness_fraction = np.diff(edges, axis=1)
    thickness_um = thickness_fraction * cortical_thickness_mm[:, None] * 1000.0
    return {
        "p01_fraction": float(np.quantile(thickness_fraction, 0.01)),
        "p05_fraction": float(np.quantile(thickness_fraction, 0.05)),
        "median_fraction": float(np.median(thickness_fraction)),
        "p01_um": float(np.quantile(thickness_um, 0.01)),
        "p05_um": float(np.quantile(thickness_um, 0.05)),
        "median_um": float(np.median(thickness_um)),
    }


def run_validation(
    atlas_edges,
    cortical_thickness_mm,
    sigmas,
    centers,
    uncertainties,
    log_sd_weights,
    n_repeats,
    seed,
):
    """Run the repeated boundary-uncertainty validation simulation."""
    rng = np.random.default_rng(seed)
    source_depth, atlas_weights = weights_for_edges(atlas_edges, n_layers=11)
    sampled_profiles = {}
    exact_atlas_profiles = {}
    atlas_estimates = {}

    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        sampled = sample_gaussian(source_depth, centers, sigma_value)
        sampled_profiles[sigma_value] = sampled
        exact_atlas_profiles[sigma_value] = exact_gaussian_laminar_mean(
            atlas_edges, centers, sigma_value
        )
        atlas_estimates[sigma_value] = reconstruct_laminar_profile(sampled, atlas_weights)

    rows = []
    confusion = {}
    for requested_rms_value in uncertainties:
        requested_rms_value = float(requested_rms_value)
        print(f"\nTarget boundary RMS = {requested_rms_value:.3f}")

        for repeat_idx in range(n_repeats):
            true_edges, achieved_rms, perturbation_scale = perturb_edges_to_target_rms(
                atlas_edges, requested_rms_value, log_sd_weights, rng
            )
            physical = physical_boundary_displacement(
                atlas_edges, true_edges, cortical_thickness_mm
            )
            rms_fraction, rms_um = boundary_specific_rms(
                atlas_edges, true_edges, cortical_thickness_mm
            )
            thickness = lamina_thickness_statistics(true_edges, cortical_thickness_mm)
            true_source_depth, true_weights = weights_for_edges(true_edges, n_layers=11)
            np.testing.assert_allclose(true_source_depth, source_depth, rtol=0.0, atol=1e-12)

            for sigma_value in sigmas:
                sigma_value = float(sigma_value)
                sampled = sampled_profiles[sigma_value]
                truth = exact_gaussian_laminar_mean(true_edges, centers, sigma_value)
                sampling_estimate = reconstruct_laminar_profile(sampled, true_weights)
                sampling_metrics = profile_metrics(sampling_estimate, truth)
                boundary_metrics = profile_metrics(exact_atlas_profiles[sigma_value], truth)
                combined_metrics = profile_metrics(atlas_estimates[sigma_value], truth)

                key = (requested_rms_value, sigma_value)
                if key not in confusion:
                    n_laminae = len(LAMINA_LABELS)
                    confusion[key] = np.zeros((n_laminae, n_laminae), dtype=np.int64)
                confusion[key] += dominant_confusion(atlas_estimates[sigma_value], truth)

                row = {
                    "requested_boundary_rms": requested_rms_value,
                    "achieved_boundary_rms": achieved_rms,
                    "boundary_rms_um": physical["rms_um"],
                    "boundary_median_abs_um": physical["median_abs_um"],
                    "boundary_q95_abs_um": physical["q95_abs_um"],
                    "boundary_max_abs_um": physical["max_abs_um"],
                    "perturbation_scale": perturbation_scale,
                    "repeat": repeat_idx,
                    "sigma": sigma_value,
                    "fwhm": 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_value,
                    "lamina_thickness_p01_fraction": thickness["p01_fraction"],
                    "lamina_thickness_p05_fraction": thickness["p05_fraction"],
                    "lamina_thickness_median_fraction": thickness["median_fraction"],
                    "lamina_thickness_p01_um": thickness["p01_um"],
                    "lamina_thickness_p05_um": thickness["p05_um"],
                    "lamina_thickness_median_um": thickness["median_um"],
                    "sampling_median_nrmse": sampling_metrics["median_nrmse"],
                    "sampling_mean_nrmse": sampling_metrics["mean_nrmse"],
                    "sampling_dominant_accuracy": sampling_metrics["dominant_accuracy"],
                    "boundary_median_nrmse": boundary_metrics["median_nrmse"],
                    "boundary_mean_nrmse": boundary_metrics["mean_nrmse"],
                    "boundary_dominant_accuracy": boundary_metrics["dominant_accuracy"],
                    "combined_median_nrmse": combined_metrics["median_nrmse"],
                    "combined_mean_nrmse": combined_metrics["mean_nrmse"],
                    "combined_dominant_accuracy": combined_metrics["dominant_accuracy"],
                }
                for boundary_idx, label in enumerate(BOUNDARY_LABELS):
                    safe_label = label.replace("/", "_")
                    row[f"rms_{safe_label}"] = rms_fraction[boundary_idx]
                    row[f"rms_{safe_label}_um"] = rms_um[boundary_idx]
                rows.append(row)

            rms_text = ", ".join(
                f"{label}={value:.0f}µm" for label, value in zip(BOUNDARY_LABELS, rms_um)
            )
            print(
                f"  repeat {repeat_idx + 1:02d}/{n_repeats}: "
                f"global={physical['rms_um']:.0f}µm; {rms_text}; "
                f"p01={thickness['p01_um']:.0f}µm; p05={thickness['p05_um']:.0f}µm",
                end="\r",
            )
        print()
    return rows, confusion


def summarize_rows(rows):
    """Aggregate repeated simulation results by uncertainty and Gaussian width."""
    summary = []
    uncertainties = sorted({float(row["requested_boundary_rms"]) for row in rows})
    sigmas = sorted({float(row["sigma"]) for row in rows})
    fields = (
        "achieved_boundary_rms",
        "boundary_rms_um",
        "boundary_median_abs_um",
        "boundary_q95_abs_um",
        "boundary_max_abs_um",
        "perturbation_scale",
        "lamina_thickness_p01_fraction",
        "lamina_thickness_p05_fraction",
        "lamina_thickness_median_fraction",
        "lamina_thickness_p01_um",
        "lamina_thickness_p05_um",
        "lamina_thickness_median_um",
        "sampling_median_nrmse",
        "sampling_mean_nrmse",
        "sampling_dominant_accuracy",
        "boundary_median_nrmse",
        "boundary_mean_nrmse",
        "boundary_dominant_accuracy",
        "combined_median_nrmse",
        "combined_mean_nrmse",
        "combined_dominant_accuracy",
    )

    for uncertainty in uncertainties:
        for sigma_value in sigmas:
            selected = [
                row
                for row in rows
                if np.isclose(row["requested_boundary_rms"], uncertainty)
                and np.isclose(row["sigma"], sigma_value)
            ]
            if not selected:
                continue

            output = {
                "requested_boundary_rms": uncertainty,
                "sigma": sigma_value,
                "fwhm": 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_value,
            }
            for field in fields:
                values = np.asarray([row[field] for row in selected], dtype=float)
                output[field] = float(np.median(values))
                output[field + "_q025"] = float(np.quantile(values, 0.025))
                output[field + "_q975"] = float(np.quantile(values, 0.975))

            for label in BOUNDARY_LABELS:
                safe_label = label.replace("/", "_")
                for suffix in ("", "_um"):
                    field = f"rms_{safe_label}{suffix}"
                    values = np.asarray([row[field] for row in selected], dtype=float)
                    output[field] = float(np.median(values))
                    output[field + "_q025"] = float(np.quantile(values, 0.025))
                    output[field + "_q975"] = float(np.quantile(values, 0.975))
            summary.append(output)
    return summary


def confusion_rows(confusion, summary):
    """Convert aggregated dominant-lamina confusion matrices to tidy rows."""
    physical_rms = {
        (float(row["requested_boundary_rms"]), float(row["sigma"])): float(
            row["boundary_rms_um"]
        )
        for row in summary
    }
    rows = []
    for (requested_rms_value, sigma_value), matrix in sorted(confusion.items()):
        for true_idx, true_label in enumerate(LAMINA_LABELS):
            total = int(matrix[true_idx].sum())
            for estimated_idx, estimated_label in enumerate(LAMINA_LABELS):
                count = int(matrix[true_idx, estimated_idx])
                rows.append(
                    {
                        "error_source": "combined",
                        "requested_boundary_rms": requested_rms_value,
                        "boundary_rms_um": physical_rms[(requested_rms_value, sigma_value)],
                        "sigma": sigma_value,
                        "fwhm": 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_value,
                        "true_lamina": true_label,
                        "estimated_lamina": estimated_label,
                        "count": count,
                        "row_fraction": count / total if total else np.nan,
                    }
                )
    return rows


def write_csv(path, rows):
    """Write a sequence of dictionaries to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_calibration(output_dir, target_profile, predicted_profile):
    """Plot requested and calibrated boundary-specific uncertainty profiles."""
    boundary_index = np.arange(len(BOUNDARY_LABELS))
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(boundary_index, target_profile, marker="o", label="Target boundary RMS profile")
    axis.plot(
        boundary_index,
        predicted_profile,
        marker="o",
        linestyle="--",
        label="Linearized prediction",
    )
    axis.set_xticks(boundary_index)
    axis.set_xticklabels(BOUNDARY_LABELS)
    axis.set_ylabel("Relative RMS uncertainty")
    axis.set_title("Calibration of log-thickness perturbation model")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_boundary_profile.png", dpi=200)
    plt.close(figure)


def plot_combined_nrmse(output_dir, summary):
    """Plot practical mapping error across boundary-uncertainty levels."""
    sigmas = sorted({row["sigma"] for row in summary})
    figure, axis = plt.subplots(figsize=(7, 5))
    for sigma_value in sigmas:
        selected = sorted(
            [row for row in summary if np.isclose(row["sigma"], sigma_value)],
            key=lambda row: row["boundary_rms_um"],
        )
        boundary_rms_um = np.asarray([row["boundary_rms_um"] for row in selected])
        metric_values = np.asarray([row["combined_median_nrmse"] for row in selected])
        lower = np.asarray([row["combined_median_nrmse_q025"] for row in selected])
        upper = np.asarray([row["combined_median_nrmse_q975"] for row in selected])
        axis.plot(boundary_rms_um, metric_values, marker="o", label=f"sigma={sigma_value:.3f}")
        axis.fill_between(boundary_rms_um, lower, upper, alpha=0.15)

    axis.set_xlabel("RMS laminar-boundary displacement (µm)")
    axis.set_ylabel("Median normalized laminar RMSE")
    axis.set_title("BigBrain-informed layer-to-lamina mapping uncertainty")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "mapping_nrmse.png", dpi=200)
    plt.close(figure)


def plot_accuracy(output_dir, summary):
    """Plot dominant-lamina recovery across boundary-uncertainty levels."""
    sigmas = sorted({row["sigma"] for row in summary})
    figure, axis = plt.subplots(figsize=(7, 5))
    for sigma_value in sigmas:
        selected = sorted(
            [row for row in summary if np.isclose(row["sigma"], sigma_value)],
            key=lambda row: row["boundary_rms_um"],
        )
        boundary_rms_um = np.asarray([row["boundary_rms_um"] for row in selected])
        metric_values = np.asarray([row["combined_dominant_accuracy"] for row in selected])
        lower = np.asarray([row["combined_dominant_accuracy_q025"] for row in selected])
        upper = np.asarray([row["combined_dominant_accuracy_q975"] for row in selected])
        axis.plot(boundary_rms_um, metric_values, marker="o", label=f"sigma={sigma_value:.3f}")
        axis.fill_between(boundary_rms_um, lower, upper, alpha=0.15)

    axis.set_xlabel("RMS laminar-boundary displacement (µm)")
    axis.set_ylabel("Dominant-lamina accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("BigBrain-informed dominant-lamina recovery")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "dominant_lamina_accuracy.png", dpi=200)
    plt.close(figure)


def plot_error_sources(output_dir, summary):
    """Plot sampling, boundary-mismatch, and combined error sources."""
    sigmas = sorted({row["sigma"] for row in summary})
    for sigma_value in sigmas:
        selected = sorted(
            [row for row in summary if np.isclose(row["sigma"], sigma_value)],
            key=lambda row: row["boundary_rms_um"],
        )
        boundary_rms_um = np.asarray([row["boundary_rms_um"] for row in selected])
        figure, axis = plt.subplots(figsize=(7, 5))
        for field, label in (
            ("sampling_median_nrmse", "Sampling only"),
            ("boundary_median_nrmse", "Boundary mismatch only"),
            ("combined_median_nrmse", "Combined"),
        ):
            metric_values = np.asarray([row[field] for row in selected])
            lower = np.asarray([row[field + "_q025"] for row in selected])
            upper = np.asarray([row[field + "_q975"] for row in selected])
            axis.plot(boundary_rms_um, metric_values, marker="o", label=label)
            axis.fill_between(boundary_rms_um, lower, upper, alpha=0.10)

        axis.set_xlabel("RMS laminar-boundary displacement (µm)")
        axis.set_ylabel("Median normalized laminar RMSE")
        axis.set_title(f"BigBrain-informed log-thickness error sources\nsigma={sigma_value:.3f}")
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / f"error_sources_sigma-{sigma_value:.3f}.png", dpi=200)
        plt.close(figure)


def plot_combined_confusion(output_dir, confusion, summary):
    """Plot row-normalized dominant-lamina confusion for the practical combined case."""
    sigmas = sorted({float(row["sigma"]) for row in summary})
    uncertainties = sorted({float(row["requested_boundary_rms"]) for row in summary})
    physical_rms = {
        (float(row["requested_boundary_rms"]), float(row["sigma"])): float(
            row["boundary_rms_um"]
        )
        for row in summary
    }
    n_columns = min(3, len(uncertainties))
    n_rows = int(np.ceil(len(uncertainties) / n_columns))

    for sigma_value in sigmas:
        figure, axes = plt.subplots(
            n_rows,
            n_columns,
            figsize=(3.6 * n_columns, 3.5 * n_rows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()
        image = None
        for axis, uncertainty in zip(axes, uncertainties):
            matrix = confusion[(uncertainty, sigma_value)].astype(float)
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_totals,
                out=np.zeros_like(matrix),
                where=row_totals > 0,
            )
            image = axis.imshow(normalized, vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
            axis.set_title(f"{physical_rms[(uncertainty, sigma_value)]:.0f} µm")
            axis.set_xticks(np.arange(len(LAMINA_LABELS)))
            axis.set_yticks(np.arange(len(LAMINA_LABELS)))
            axis.set_xticklabels(LAMINA_LABELS)
            axis.set_yticklabels(LAMINA_LABELS)

        for axis in axes[len(uncertainties):]:
            axis.set_visible(False)
        for axis in axes[-n_columns:]:
            if axis.get_visible():
                axis.set_xlabel("Estimated lamina")
        for axis in axes[::n_columns]:
            if axis.get_visible():
                axis.set_ylabel("True lamina")

        figure.suptitle(f"Combined dominant-lamina confusion, sigma={sigma_value:.3f}")
        if image is not None:
            visible_axes = [axis for axis in axes if axis.get_visible()]
            colorbar = figure.colorbar(image, ax=visible_axes, fraction=0.025, pad=0.02)
            colorbar.set_label("Fraction within true lamina")
        figure.subplots_adjust(
            left=0.08, right=0.89, bottom=0.08, top=0.90, wspace=0.18, hspace=0.25
        )
        filename = f"combined_confusion_sigma-{sigma_value:.3f}.png"
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)


def plot_boundary_specific_rms(output_dir, summary):
    """Plot realized RMS displacement for each internal laminar boundary."""
    sigma_value = min(row["sigma"] for row in summary)
    selected = sorted(
        [row for row in summary if np.isclose(row["sigma"], sigma_value)],
        key=lambda row: row["boundary_rms_um"],
    )
    boundary_rms_um = np.asarray([row["boundary_rms_um"] for row in selected])
    figure, axis = plt.subplots(figsize=(7, 5))
    for label in BOUNDARY_LABELS:
        safe_label = label.replace("/", "_")
        metric_values = np.asarray([row[f"rms_{safe_label}_um"] for row in selected])
        axis.plot(boundary_rms_um, metric_values, marker="o", label=label)
    axis.plot(boundary_rms_um, boundary_rms_um, linestyle="--", linewidth=1.0, label="Global RMS")
    axis.set_xlabel("Global RMS boundary displacement (µm)")
    axis.set_ylabel("Boundary-specific RMS displacement (µm)")
    axis.set_title("Realized BigBrain-informed uncertainty profile")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "boundary_specific_rms.png", dpi=200)
    plt.close(figure)


def plot_lamina_thickness(output_dir, summary):
    """Plot robust laminar-thickness diagnostics across uncertainty levels."""
    sigma_value = min(row["sigma"] for row in summary)
    selected = sorted(
        [row for row in summary if np.isclose(row["sigma"], sigma_value)],
        key=lambda row: row["boundary_rms_um"],
    )
    boundary_rms_um = np.asarray([row["boundary_rms_um"] for row in selected])
    p01 = np.asarray([row["lamina_thickness_p01_um"] for row in selected])
    p05 = np.asarray([row["lamina_thickness_p05_um"] for row in selected])
    median = np.asarray([row["lamina_thickness_median_um"] for row in selected])

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(boundary_rms_um, p01, marker="o", label="1st percentile")
    axis.plot(boundary_rms_um, p05, marker="o", label="5th percentile")
    axis.plot(boundary_rms_um, median, marker="o", label="Median")
    axis.set_xlabel("Global RMS boundary displacement (µm)")
    axis.set_ylabel("True lamina thickness (µm)")
    axis.set_title("Anatomical consequences of log-thickness perturbation")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "lamina_thickness_vs_uncertainty.png", dpi=200)
    plt.close(figure)


def print_calibration(target_profile, predicted_profile, log_sd_weights, method):
    """Print the perturbation-model calibration."""
    print("\nPerturbation calibration")
    print("========================")
    print(f"Method: {method}")
    print("\nTarget and predicted boundary RMS profiles:")
    for label, target, predicted in zip(BOUNDARY_LABELS, target_profile, predicted_profile):
        print(f"  {label:7s}: target={target:.3f}, predicted={predicted:.3f}")
    print("\nDerived log-thickness SD weights:")
    for label, weight in zip(LAMINA_LABELS, log_sd_weights):
        print(f"  {label:3s}: {weight:.3f}")


def print_summary(summary):
    """Print a concise summary of the final validation results."""
    uncertainties = sorted({row["requested_boundary_rms"] for row in summary})
    sigmas = sorted({row["sigma"] for row in summary})
    print("\nLayer-to-lamina mapping validation summary")
    print("==========================================")
    for sigma_value in sigmas:
        print(f"\nsigma={sigma_value:.3f}")
        for uncertainty in uncertainties:
            matches = [
                row
                for row in summary
                if np.isclose(row["sigma"], sigma_value)
                and np.isclose(row["requested_boundary_rms"], uncertainty)
            ]
            if not matches:
                continue
            row = matches[0]
            print(
                f"  target={uncertainty:.3f}, physical={row['boundary_rms_um']:.1f} µm, "
                f"combined NRMSE={row['combined_median_nrmse']:.4f}, "
                f"accuracy={row['combined_dominant_accuracy']:.3f}, "
                f"p01={row['lamina_thickness_p01_um']:.1f} µm, "
                f"p05={row['lamina_thickness_p05_um']:.1f} µm"
            )


def main():
    """Run the complete BigBrain-informed layer-to-lamina validation."""
    args = parse_args()
    validate_args(args)
    subjects_dir = resolve_subjects_dir(args.subjects_dir)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not subjects_dir.is_dir():
        raise FileNotFoundError(f"SUBJECTS_DIR not found: {subjects_dir}")

    sigmas = np.asarray(args.sigmas, dtype=float)
    uncertainties = np.asarray(args.boundary_rms, dtype=float)
    boundary_weights = np.asarray(args.boundary_weights, dtype=float)
    centers = np.linspace(0.0, 1.0, args.n_centers)
    surf_set = LayerSurfaceSet(args.subject, 11, subjects_dir=str(subjects_dir))
    candidates, total_columns = choose_candidate_columns(surf_set, args.n_columns, args.seed)
    valid_columns, atlas_edges = get_atlas_edges(surf_set, candidates, args.n_columns)

    all_thickness = np.asarray(surf_set.get_cortical_thickness(stage="ds"), dtype=float)
    cortical_thickness_mm = all_thickness[valid_columns]
    valid = np.isfinite(cortical_thickness_mm) & (cortical_thickness_mm > 0)
    valid_columns = valid_columns[valid]
    atlas_edges = atlas_edges[valid]
    cortical_thickness_mm = cortical_thickness_mm[valid]
    if len(valid_columns) == 0:
        raise RuntimeError("No valid cortical columns remain after thickness filtering.")

    calibration = calibrate_log_thickness_weights(atlas_edges, boundary_weights)
    log_sd_weights, target_profile, predicted_profile = calibration[:3]
    calibration_matrix, calibration_method = calibration[3:]

    print(f"Subject: {args.subject}")
    print(f"SUBJECTS_DIR: {subjects_dir}")
    print(f"Total downsampled cortical columns: {total_columns}")
    print(f"Valid columns analysed: {len(valid_columns)}")
    print(f"Median cortical thickness: {np.median(cortical_thickness_mm):.3f} mm")
    print_calibration(target_profile, predicted_profile, log_sd_weights, calibration_method)

    rows, confusion = run_validation(
        atlas_edges,
        cortical_thickness_mm,
        sigmas,
        centers,
        uncertainties,
        log_sd_weights,
        args.n_repeats,
        args.seed,
    )
    summary = summarize_rows(rows)
    confusion_table = confusion_rows(confusion, summary)
    write_csv(output_dir / "layer_to_lamina_mapping_repeats.csv", rows)
    write_csv(output_dir / "layer_to_lamina_mapping_summary.csv", summary)
    write_csv(output_dir / "layer_to_lamina_mapping_confusion.csv", confusion_table)

    np.savez_compressed(
        output_dir / "layer_to_lamina_mapping_geometry.npz",
        columns=valid_columns,
        atlas_edges=atlas_edges,
        cortical_thickness_mm=cortical_thickness_mm,
        sigmas=sigmas,
        centers=centers,
        boundary_rms=uncertainties,
        requested_boundary_weights=boundary_weights,
        target_boundary_profile=target_profile,
        predicted_boundary_profile=predicted_profile,
        log_thickness_sd_weights=log_sd_weights,
        calibration_matrix=calibration_matrix,
    )

    plot_calibration(output_dir, target_profile, predicted_profile)
    plot_combined_nrmse(output_dir, summary)
    plot_accuracy(output_dir, summary)
    plot_error_sources(output_dir, summary)
    plot_combined_confusion(output_dir, confusion, summary)
    plot_boundary_specific_rms(output_dir, summary)
    plot_lamina_thickness(output_dir, summary)
    print_summary(summary)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
