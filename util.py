import json
import matlab.engine
from contextlib import contextmanager

@contextmanager
def matlab_context(eng):
    # Start MATLAB engine
    close_matlab=False
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.addpath('./matlab', nargout=0)
        close_matlab=True
    try:
        yield eng
    finally:
        # Close MATLAB engine
        if close_matlab:
            eng.quit()


def get_spm_path():
    with open('settings.json') as settings_file:
        parameters = json.load(settings_file)
    spm_path = parameters["spm_path"]
    return spm_path