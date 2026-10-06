# Contributing to laMEG

Contributions to laMEG are welcome, including bug fixes, documentation improvements, tests, new functionality, and methodological validation.

## Reporting bugs and requesting features

Please open a GitHub issue before starting substantial changes. Describe:

- the problem or proposed feature;
- the expected behaviour;
- the current behaviour, where applicable;
- a minimal example or dataset needed to reproduce the problem;
- the operating system and laMEG version or commit.

For methodological changes, please also describe the scientific motivation and how the proposed change will be validated.

## Development setup

Clone the repository and install laMEG in editable mode:

```bash
git clone https://github.com/danclab/laMEG.git
cd laMEG
pip install -e .
```

Install the development requirements:

```bash
pip install -r dev_requirements.txt
```

SPM-dependent functionality also requires the laMEG post-installation step:

```bash
lameg-postinstall
```

## Tests

Changes should include tests where appropriate.

Run the test suite with:

```bash
pytest
```

Run linting with:

```bash
pylint --rcfile=.pylintrc $(git ls-files '*.py')
```

New scientific or numerical functionality should include tests of expected numerical behaviour in addition to basic execution tests.

## Pull requests

Pull requests should:

1. address a clearly defined issue or change;
2. keep unrelated changes separate;
3. include tests for new or modified behaviour;
4. update public API documentation when required;
5. avoid unnecessary changes to existing public interfaces;
6. explain any scientifically meaningful change in behaviour.

Changes to established public APIs should preserve backward compatibility where practical. Breaking changes should be discussed before implementation and documented in the release notes.

## Scientific validation

laMEG implements methods for depth-resolved inference in an ill-posed inverse problem. Changes that affect forward modelling, source inversion, cortical geometry, depth mapping, model comparison, or quantitative interpretation therefore require appropriate scientific validation.

Validation analyses that support the software but are not part of the public library API may be maintained under `validation/`.

## Code style

Follow the style of the existing codebase. Public functions and classes should have NumPy-style docstrings.

Prefer small, testable functions and explicit metadata over implicit assumptions about source ordering, geometry, coordinate systems, or units.

## Development roadmap

The current development roadmap is maintained on the laMEG GitHub wiki. Major proposed changes should be consistent with that roadmap or discussed through a GitHub issue before implementation.

## Licensing

laMEG is distributed under the GNU General Public License v3.0. By submitting a contribution, you agree that your contribution may be distributed under the same license.

## Citation

If you use laMEG in research, please cite the software and the methodological publication(s) relevant to the analyses you use. Citation metadata are provided in `CITATION.cff`.