"""Validate finite-surface sampling for BigBrain layer-to-lamina mapping.

A continuous Gaussian cortical-depth profile is sampled on a finite set of equidistant laMEG
surfaces, transformed to BigBrain-defined laminae, and compared with the exact continuous laminar
means. The validation isolates the approximation introduced by finite cortical-depth sampling.
"""

# pylint: disable=duplicate-code

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf  # pylint: disable=no-name-in-module

from lameg.laminar import compute_bigbrain_laminar_weights, surface_to_laminae
from lameg.surf import LayerSurfaceSet

LAMINA_LABELS = ("I", "II", "III", "IV", "V", "VI")
DEFAULT_LAYER_COUNTS = (2, 6, 11, 15, 21)
DEFAULT_SIGMAS = (0.025, 0.05, 0.10, 0.20)


class CachedBoundarySurfaceSet:  # pylint: disable=too-few-public-methods
    """Minimal surface-set interface backed by cached BigBrain boundaries."""

    def __init__(self, n_layers, edges):
        """Store layer spacing and cached cumulative boundaries."""
        self.n_layers = int(n_layers)
        self.layer_spacing = np.linspace(1.0, 0.0, self.n_layers)
        self._boundaries = np.asarray(edges[:, 1:], dtype=float).T

    def get_bigbrain_layer_boundaries(self, subj_coord=None):
        """Return cached BigBrain boundaries."""
        if subj_coord is not None:
            raise ValueError("Cached boundaries do not accept `subj_coord`.")
        return self._boundaries


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate finite-surface sampling for BigBrain layer-to-lamina mapping."
    )
    parser.add_argument("--subject", default="sub-104")
    parser.add_argument("--subjects-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-columns", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layer-counts", type=int, nargs="+", default=list(DEFAULT_LAYER_COUNTS)
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument("--n-centers", type=int, default=101)
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
    return (repo_root() / "validation" / "bigbrain_mapping" / "sampling_output").resolve()


def validate_args(args):
    """Validate command-line arguments."""
    if args.n_columns < 0:
        raise ValueError("--n-columns must be >= 0.")
    if args.n_centers < 2:
        raise ValueError("--n-centers must be >= 2.")
    if any(layer_count < 2 for layer_count in args.layer_counts):
        raise ValueError("All layer counts must be >= 2.")
    if len(set(args.layer_counts)) != len(args.layer_counts):
        raise ValueError("--layer-counts must not contain duplicates.")

    sigmas = np.asarray(args.sigmas, dtype=float)
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("All sigma values must be finite and > 0.")


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


def get_reference_edges(surf_set, candidate_columns, requested):
    """Map BigBrain boundaries once and retain valid cortical columns."""
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


def weights_for_layer_count(n_layers, edges):
    """Compute mapping weights at one reconstructed-surface density."""
    proxy = CachedBoundarySurfaceSet(n_layers, edges)
    returned_edges, weights = compute_bigbrain_laminar_weights(proxy)
    np.testing.assert_allclose(returned_edges, edges, rtol=0.0, atol=1e-12)
    source_depth = 1.0 - np.asarray(proxy.layer_spacing, dtype=float)
    return source_depth, np.asarray(weights, dtype=float)


def sample_gaussian(source_depth, centers, sigma):
    """Sample Gaussian cortical-depth profiles at reconstructed surfaces."""
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


def dominant_confusion(true_dominant, estimated_dominant):
    """Return a row-normalized dominant-lamina confusion matrix."""
    n_laminae = len(LAMINA_LABELS)
    encoded = true_dominant.ravel() * n_laminae + estimated_dominant.ravel()
    matrix = np.bincount(encoded, minlength=n_laminae**2).reshape(n_laminae, n_laminae)
    matrix = matrix.astype(float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def evaluate_mapping(edges, weights, source_depth, centers, sigma):
    """Evaluate one layer-count and Gaussian-width combination."""
    n_columns = edges.shape[0]
    sampled = sample_gaussian(source_depth, centers, sigma)
    layer_data = np.broadcast_to(
        sampled[:, None, :], (len(source_depth), n_columns, len(centers))
    )
    estimated = np.transpose(surface_to_laminae(layer_data, weights), (1, 0, 2))
    truth = exact_gaussian_laminar_mean(edges, centers, sigma)

    error = estimated - truth
    abs_error = np.abs(error)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    truth_scale = np.maximum(np.max(np.abs(truth), axis=1), np.finfo(float).eps)
    nrmse = rmse / truth_scale
    true_dominant = np.argmax(truth, axis=1)
    estimated_dominant = np.argmax(estimated, axis=1)

    return {
        "median_nrmse_by_depth": np.median(nrmse, axis=0),
        "q25_nrmse_by_depth": np.quantile(nrmse, 0.25, axis=0),
        "q75_nrmse_by_depth": np.quantile(nrmse, 0.75, axis=0),
        "median_rmse_by_depth": np.median(rmse, axis=0),
        "median_max_abs_error_by_depth": np.median(np.max(abs_error, axis=1), axis=0),
        "accuracy_by_depth": np.mean(estimated_dominant == true_dominant, axis=0),
        "confusion": dominant_confusion(true_dominant, estimated_dominant),
        "lamina_abs_error": np.median(abs_error, axis=(0, 2)),
    }


def run_validation(edges, layer_counts, sigmas, centers):
    """Run finite-surface sampling validation for all requested configurations."""
    results = {}
    for n_layers in layer_counts:
        source_depth, weights = weights_for_layer_count(n_layers, edges)
        for sigma_value in sigmas:
            sigma_value = float(sigma_value)
            print(f"Evaluating N={n_layers}, sigma={sigma_value:.3f}")
            results[(n_layers, sigma_value)] = evaluate_mapping(
                edges, weights, source_depth, centers, sigma_value
            )
    return results


def build_summary_rows(results, layer_counts, sigmas):
    """Build one global summary row per layer-count and Gaussian width."""
    rows = []
    for n_layers in layer_counts:
        for sigma_value in sigmas:
            result = results[(n_layers, float(sigma_value))]
            rows.append(
                {
                    "n_layers": n_layers,
                    "sigma": float(sigma_value),
                    "fwhm": 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(sigma_value),
                    "median_nrmse_over_depth": float(
                        np.median(result["median_nrmse_by_depth"])
                    ),
                    "worst_median_nrmse_over_depth": float(
                        np.max(result["median_nrmse_by_depth"])
                    ),
                    "median_rmse_over_depth": float(
                        np.median(result["median_rmse_by_depth"])
                    ),
                    "worst_median_abs_error_over_depth": float(
                        np.max(result["median_max_abs_error_by_depth"])
                    ),
                    "mean_dominant_lamina_accuracy": float(
                        np.mean(result["accuracy_by_depth"])
                    ),
                }
            )
    return rows


def build_depth_rows(results, layer_counts, sigmas, centers):
    """Build depth-resolved summary rows."""
    rows = []
    for n_layers in layer_counts:
        for sigma_value in sigmas:
            sigma_value = float(sigma_value)
            result = results[(n_layers, sigma_value)]
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_value
            for center_idx, center in enumerate(centers):
                rows.append(
                    {
                        "n_layers": n_layers,
                        "sigma": sigma_value,
                        "fwhm": fwhm,
                        "center_depth": float(center),
                        "median_nrmse": float(result["median_nrmse_by_depth"][center_idx]),
                        "q25_nrmse": float(result["q25_nrmse_by_depth"][center_idx]),
                        "q75_nrmse": float(result["q75_nrmse_by_depth"][center_idx]),
                        "median_rmse": float(result["median_rmse_by_depth"][center_idx]),
                        "median_max_abs_error": float(
                            result["median_max_abs_error_by_depth"][center_idx]
                        ),
                        "dominant_lamina_accuracy": float(
                            result["accuracy_by_depth"][center_idx]
                        ),
                    }
                )
    return rows


def write_csv(path, rows):
    """Write a sequence of dictionaries to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_nrmse_vs_layer_count(output_dir, results, layer_counts, sigmas):
    """Plot overall normalized RMSE versus reconstructed surface count."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        metric_values = [
            np.median(results[(n_layers, sigma_value)]["median_nrmse_by_depth"])
            for n_layers in layer_counts
        ]
        axis.plot(layer_counts, metric_values, marker="o", label=f"sigma={sigma_value:.3f}")
    axis.set_xlabel("Number of reconstructed surfaces")
    axis.set_ylabel("Median normalized laminar RMSE")
    axis.set_xticks(layer_counts)
    axis.set_title("Finite-depth sampling error")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "nrmse_vs_layer_count.png", dpi=200)
    plt.close(figure)


def plot_accuracy_vs_layer_count(output_dir, results, layer_counts, sigmas):
    """Plot dominant-lamina accuracy versus reconstructed surface count."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        metric_values = [
            np.mean(results[(n_layers, sigma_value)]["accuracy_by_depth"])
            for n_layers in layer_counts
        ]
        axis.plot(layer_counts, metric_values, marker="o", label=f"sigma={sigma_value:.3f}")
    axis.set_xlabel("Number of reconstructed surfaces")
    axis.set_ylabel("Dominant-lamina accuracy")
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(layer_counts)
    axis.set_title("Recovery of dominant BigBrain lamina")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "dominant_lamina_accuracy_vs_layer_count.png", dpi=200)
    plt.close(figure)


def plot_error_vs_depth(output_dir, results, centers, edges, sigmas, n_layers):
    """Plot depth-dependent normalized RMSE for one representative layer count."""
    median_boundaries = np.median(edges[:, 1:-1], axis=0)
    figure, axis = plt.subplots(figsize=(8, 5))
    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        axis.plot(
            centers,
            results[(n_layers, sigma_value)]["median_nrmse_by_depth"],
            label=f"sigma={sigma_value:.3f}",
        )
    for boundary in median_boundaries:
        axis.axvline(boundary, linestyle="--", linewidth=0.8, alpha=0.45)
    axis.set_xlabel("Gaussian peak depth (0=pial, 1=white)")
    axis.set_ylabel("Median normalized laminar RMSE")
    axis.set_title(f"Depth dependence of sampling error, N={n_layers}")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / f"nrmse_vs_depth_n{n_layers}.png", dpi=200)
    plt.close(figure)


def plot_error_vs_lamina_thickness(output_dir, results, edges, sigmas, n_layers):
    """Plot laminar error against median BigBrain lamina thickness."""
    median_thickness = np.median(np.diff(edges, axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(7, 5))
    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        error = results[(n_layers, sigma_value)]["lamina_abs_error"]
        axis.plot(median_thickness, error, marker="o", label=f"sigma={sigma_value:.3f}")
    axis.set_xlabel("Median BigBrain lamina thickness (fraction of cortical thickness)")
    axis.set_ylabel("Median absolute laminar error")
    axis.set_title(f"Sampling error versus lamina thickness, N={n_layers}")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / f"lamina_error_vs_thickness_n{n_layers}.png", dpi=200)
    plt.close(figure)


def plot_confusion_matrices(output_dir, results, sigmas, n_layers):
    """Save one row-normalized dominant-lamina confusion matrix per Gaussian width."""
    for sigma_value in sigmas:
        sigma_value = float(sigma_value)
        matrix = results[(n_layers, sigma_value)]["confusion"]
        figure, axis = plt.subplots(figsize=(5, 5))
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(len(LAMINA_LABELS)))
        axis.set_yticks(np.arange(len(LAMINA_LABELS)))
        axis.set_xticklabels(LAMINA_LABELS)
        axis.set_yticklabels(LAMINA_LABELS)
        axis.set_xlabel("Estimated dominant lamina")
        axis.set_ylabel("True dominant lamina")
        axis.set_title(f"Dominant-lamina recovery, N={n_layers}, sigma={sigma_value:.3f}")
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Proportion")
        figure.tight_layout()
        filename = f"confusion_n{n_layers}_sigma-{sigma_value:.3f}.png"
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)


def save_results(path, results, columns, edges, layer_counts, sigmas, centers):
    """Save numerical validation arrays in a compressed NPZ archive."""
    payload = {
        "columns": columns,
        "edges": edges,
        "layer_counts": np.asarray(layer_counts, dtype=int),
        "sigmas": np.asarray(sigmas, dtype=float),
        "centers": centers,
    }
    for n_layers in layer_counts:
        for sigma_value in sigmas:
            sigma_value = float(sigma_value)
            prefix = f"n{n_layers}_sigma-{sigma_value:.3f}".replace(".", "p")
            result = results[(n_layers, sigma_value)]
            for field, value in result.items():
                payload[f"{prefix}_{field}"] = value
    np.savez_compressed(path, **payload)


def print_summary(summary_rows):
    """Print concise global sampling-validation results."""
    print("\nFinite-depth sampling summary")
    print("=============================")
    for row in summary_rows:
        print(
            f"N={row['n_layers']:2d}, sigma={row['sigma']:.3f}: "
            f"median NRMSE={row['median_nrmse_over_depth']:.4f}, "
            f"accuracy={row['mean_dominant_lamina_accuracy']:.3f}"
        )


def main():
    """Run the complete finite-depth sampling validation."""
    args = parse_args()
    validate_args(args)
    subjects_dir = resolve_subjects_dir(args.subjects_dir)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not subjects_dir.is_dir():
        raise FileNotFoundError(f"SUBJECTS_DIR not found: {subjects_dir}")

    layer_counts = [int(value) for value in args.layer_counts]
    sigmas = np.asarray(args.sigmas, dtype=float)
    centers = np.linspace(0.0, 1.0, args.n_centers)
    surf_set = LayerSurfaceSet(args.subject, 11, subjects_dir=str(subjects_dir))
    candidate_columns, total_columns = choose_candidate_columns(
        surf_set, args.n_columns, args.seed
    )
    valid_columns, edges = get_reference_edges(surf_set, candidate_columns, args.n_columns)

    print(f"Subject: {args.subject}")
    print(f"SUBJECTS_DIR: {subjects_dir}")
    print(f"Total downsampled cortical columns: {total_columns}")
    print(f"Valid BigBrain columns analysed: {len(valid_columns)}")
    print(f"Layer counts: {layer_counts}")
    print(f"Gaussian sigmas: {sigmas.tolist()}")

    results = run_validation(edges, layer_counts, sigmas, centers)
    summary_rows = build_summary_rows(results, layer_counts, sigmas)
    depth_rows = build_depth_rows(results, layer_counts, sigmas, centers)
    write_csv(output_dir / "global_summary.csv", summary_rows)
    write_csv(output_dir / "depth_summary.csv", depth_rows)
    save_results(
        output_dir / "sampling_validation_results.npz",
        results,
        valid_columns,
        edges,
        layer_counts,
        sigmas,
        centers,
    )

    plot_nrmse_vs_layer_count(output_dir, results, layer_counts, sigmas)
    plot_accuracy_vs_layer_count(output_dir, results, layer_counts, sigmas)
    representative_n = 11 if 11 in layer_counts else layer_counts[len(layer_counts) // 2]
    plot_error_vs_depth(output_dir, results, centers, edges, sigmas, representative_n)
    plot_error_vs_lamina_thickness(output_dir, results, edges, sigmas, representative_n)
    plot_confusion_matrices(output_dir, results, sigmas, representative_n)
    print_summary(summary_rows)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
