# pylint: disable=missing-module-docstring
import os
import importlib

# ----------------------------------------------------------------------
# Check whether postinstall has been completed
# ----------------------------------------------------------------------
_marker = os.path.join(os.path.expanduser("~"), ".lameg_postinstall")
if not os.path.exists(_marker):
    print(
        "\nPlease run `lameg-postinstall` to complete setup "
        "(Matlab runtime and SPM installation and Jupyter configuration).\n"
    )

# ----------------------------------------------------------------------
# Lazy import of submodules (prevents circular import before SPM install)
# ----------------------------------------------------------------------
__all__ = ["invert", "laminar", "surf", "util", "viz"]

# Load all submodules eagerly when building docs
if os.environ.get("SPHINX_BUILD") == "1":
    for mod_name in __all__:
        globals()[mod_name] = importlib.import_module(f"lameg.{mod_name}")
else:
    def __getattr__(name):
        if name in __all__:
            return importlib.import_module(f"lameg.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
