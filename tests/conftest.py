"""
This module creates an spm instance fixture so other tests can use it
"""

import pytest
import spm_standalone

from tests.source_test_utils import make_source_file


@pytest.fixture(scope="session", autouse=True)
def matlab_runtime():
    """Initialize MATLAB Runtime without the JVM for automated tests."""
    spm_standalone.initialize_runtime(["-nojvm"])


@pytest.fixture(scope="session")
def spm():
    """
    A pytest fixture that initializes a shared instance of the SPM standalone interface for use
    across all tests in a session.

    This fixture is designed to initialize the SPM software only once per test session, reducing
    initialization overhead and ensuring consistency across tests. It is scoped at the session
    level, meaning the same instance is reused in all tests requiring it within the same test
    execution session.

    Yields:
        spm_instance (SPM): An initialized SPM object ready for use.

    After all tests have completed, the `terminate` method is called to properly close any
    resources or processes started by the SPM instance.

    Example:
        def test_spm_processing(spm):
            # Use 'spm' to perform some operations
            result = spm.some_spm_function()
            assert result is not None
    """
    spm_instance = spm_standalone.initialize()
    yield spm_instance
    spm_instance.terminate()


@pytest.fixture
def source_file(tmp_path):
    """Source file without an explicit trial dimension."""
    fname = tmp_path / "source.h5"
    expected = make_source_file(
        fname,
        with_trials=False,
    )
    return fname, expected


@pytest.fixture
def trial_source_file(tmp_path):
    """Source file with an explicit trial dimension."""
    fname = tmp_path / "source_trials.h5"
    expected = make_source_file(
        fname,
        with_trials=True,
    )
    return fname, expected
