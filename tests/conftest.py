"""
This module creates an spm instance fixture so other tests can use it
"""

import pytest
import spm_standalone

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
