# BigBrain mapping validation

This directory contains validation analyses for laMEG's mapping from
reconstructed cortical-depth surfaces to BigBrain-defined cortical laminae.

These scripts are intended to quantify the numerical accuracy and robustness 
of the layer-to-lamina transformation under realistic analysis conditions using 
the test subject.

## 1. Finite-depth sampling

`validate_sampling.py` tests the approximation of a continuous
cortical-depth activity profile by a finite number of equidistant
surfaces.

The steps are:

- obtain subject-specific BigBrain laminar boundaries for cortical columns
- generate continuous Gaussian depth profiles
- sample those profiles using different numbers of surfaces
- apply the laMEG layer-to-lamina mapping
- compare the estimated laminar means with exact analytic continuous means

The default surface counts are:

- 2
- 6
- 11
- 15
- 21

The default Gaussian widths are:

- sigma = 0.025
- sigma = 0.050
- sigma = 0.100
- sigma = 0.200

This analysis quantifies the error caused specifically by finite depth sampling.

Run from the repository root:

```bash
python validation/bigbrain_mapping/validate_sampling.py
```

Outputs are written to:

```text
validation/bigbrain_mapping/results/sampling/
```

The principal outputs are:

- `global_summary.csv`
- `depth_summary.csv`
- `sampling_validation_results.npz`
- `nrmse_vs_layer_count.png`
- `dominant_lamina_accuracy_vs_layer_count.png`
- depth-dependent error plots
- dominant-lamina confusion matrices

## 2. Robustness to BigBrain boundary uncertainty

`validate_layer_to_lamina_mapping.py` evaluates robustness of the practical
11-surface layer-to-lamina transform to uncertainty in the BigBrain-derived
laminar boundaries.

Rather than perturbing the five internal boundaries independently, the simulation
perturbs the six positive laminar thicknesses in log space and renormalizes them
to total cortical thickness. This guarantees that:

- all laminae remain positive
- laminar ordering is preserved
- the pial boundary remains at 0
- the white-matter boundary remains at 1
- boundary crossings cannot occur

The perturbation model uses the qualitative relative boundary uncertainty profile:

| Boundary | Relative uncertainty |
| --- | ---: |
| I/II | 0.7 |
| II/III | 1.0 |
| III/IV | 1.4 |
| IV/V | 1.4 |
| V/VI | 1.3 |

These values specify a relative uncertainty profile, not empirically measured
absolute BigBrain-to-subject errors.

The default global RMS boundary-displacement conditions are:

- 0%
- 1%
- 2.5%
- 5%
- 7.5%
- 10% of cortical thickness

For the standard test subject these correspond approximately to increasing
physical boundary displacements from 0 to about 320 micrometres. The highest
condition should be interpreted as a severe stress test rather than as an estimate
of expected anatomical uncertainty.

The analysis compares three error sources:

1. **Sampling only**  
   The perturbed boundaries are treated as known and the continuous profile is
   represented using 11 reconstructed surfaces.

2. **Boundary mismatch only**  
   Continuous profiles are evaluated exactly, but the unperturbed BigBrain
   boundaries are used instead of the perturbed boundaries.

3. **Combined**  
   The profile is represented using 11 reconstructed surfaces and mapped using
   the unperturbed BigBrain boundaries. This corresponds to the practical laMEG
   analysis case.

These are error sources rather than an additive error decomposition; their effects
can partially cancel.

Run the full validation from the repository root:

```bash
python validation/bigbrain_mapping/validate_layer_to_lamina_mapping.py \
    --n-columns 5000 --n-repeats 50
```

Outputs are written to:

```text
validation/bigbrain_mapping/results/layer_to_lamina_mapping/
```

Principal outputs include:

- `layer_to_lamina_mapping_summary.csv`
- `layer_to_lamina_mapping_repeats.csv`
- `layer_to_lamina_mapping_confusion.csv`
- `layer_to_lamina_mapping_geometry.npz`
- `mapping_nrmse.png`
- `dominant_lamina_accuracy.png`
- `error_sources_sigma-*.png`
- `combined_confusion_sigma-*.png`
- `boundary_specific_rms.png`
- `lamina_thickness_vs_uncertainty.png`
- `calibration_boundary_profile.png`

## Interpretation

Together, the two analyses answer different questions.

`validate_sampling.py` asks:

> How accurately can a continuous cortical-depth profile be represented using a
> finite number of reconstructed surfaces?

`validate_layer_to_lamina_mapping.py` asks:

> Given the standard 11-surface representation, how robust is laminar inference
> to uncertainty in the BigBrain-derived laminar boundaries?

The current results show that 11 surfaces reproduce broad cortical-depth profiles
accurately, while extremely narrow profiles remain limited by finite sampling.
For broader profiles, uncertainty in the laminar boundaries becomes the dominant
source of error as boundary displacement increases.

Dominant-lamina errors under boundary perturbation are predominantly between
adjacent laminae rather than large jumps across cortical depth.

### Choice of surface count

The sampling analysis supports the use of 11 reconstructed surfaces as a practical
default. At this resolution, sampling error is already small for moderately broad
depth profiles and becomes negligible for broader profiles. Increasing the number
of surfaces mainly improves recovery of extremely narrow profiles, whose spatial
extent is smaller than the spacing between the 11 reconstructed surfaces.

For broader profiles, uncertainty in the BigBrain-derived laminar boundaries
becomes comparable to or larger than the residual sampling error at 11 surfaces.
Increasing the number of reconstructed surfaces therefore provides diminishing
returns for typical laminar inference and cannot compensate for anatomical
boundary uncertainty.

laMEG therefore uses 11 surfaces as a reasonable accuracy/complexity trade-off.
Higher surface counts may still be useful for analyses specifically targeting very
focal cortical-depth profiles, in which case a surface-count sensitivity analysis
is recommended.