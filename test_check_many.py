import importlib.util
from pathlib import Path
import sys
import types

# Create lightweight stubs for optional heavy dependencies used in util.py
sys.modules.setdefault("mne", types.ModuleType("mne"))
sys.modules.setdefault("mne.coreg", types.ModuleType("mne.coreg"))
sys.modules["mne.coreg"].Coregistration = object
sys.modules.setdefault("mne.io", types.ModuleType("mne.io"))
sys.modules["mne.io"]._empty_info = lambda: None
sys.modules.setdefault("mne.transforms", types.ModuleType("mne.transforms"))
sys.modules["mne.transforms"].apply_trans = lambda *a, **k: None
sys.modules["mne.transforms"].invert_transform = lambda *a, **k: None
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("h5py", types.ModuleType("h5py"))
sys.modules["h5py"].File = lambda *a, **k: (_ for _ in ()).throw(ModuleNotFoundError("h5py"))
sys.modules.setdefault("nibabel", types.ModuleType("nibabel"))
class _Img:
    def agg_data(self):
        return []
sys.modules["nibabel"].load = lambda *a, **k: _Img()
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("vtk", types.ModuleType("vtk"))
sys.modules.setdefault("spm_standalone", types.ModuleType("spm_standalone"))
sys.modules.setdefault("scipy.io", types.ModuleType("scipy.io"))
sys.modules["scipy.io"].savemat = lambda *a, **k: None
sys.modules["scipy.io"].loadmat = lambda *a, **k: {}
sys.modules.setdefault("scipy.spatial", types.ModuleType("scipy.spatial"))
sys.modules["scipy.spatial"].KDTree = object
sys.modules.setdefault("scipy.stats", types.ModuleType("scipy.stats"))
sys.modules["scipy.stats"].t = object

# Import util module without triggering heavy package imports
spec = importlib.util.spec_from_file_location(
    "util", Path(__file__).parent / "lameg" / "util.py"
)
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

check_many = util.check_many
get_files = util.get_files


def test_get_files_without_strings(tmp_path):
    (tmp_path / 'a.txt').write_text('')
    (tmp_path / 'b.txt').write_text('')
    files_any = get_files(tmp_path, '*.txt', check='any')
    files_all = get_files(tmp_path, '*.txt')
    assert len(files_any) == 2
    assert len(files_all) == 2
    assert check_many([], 'anything', 'any')
    assert check_many([], 'anything', 'all')
