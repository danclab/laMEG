"""
laMEG Post-installation Script
------------------------------

This module performs all post-installation setup tasks for the laMEG package.
It installs the DANC SPM Python interface, configures Jupyter notebook
extensions (e.g., `k3d`), and creates a marker file in the user's home
directory (`~/.lameg_postinstall`) indicating that setup has been completed.

Typical usage (after `pip install lameg`):

    $ lameg-postinstall

If laMEG is installed inside a conda environment, users should deactivate and
reactivate the environment after running this script so that any new environment
variables take effect:

    conda deactivate
    conda activate <env_name>
"""

import logging
import os
import subprocess
import sys
import tempfile

DANC_SPM_VERSION = "v0.1.0"
DANC_SPM_REPO = "https://github.com/danclab/DANC_spm_python.git"

# Set up logging to both the console and a log file in the user's home directory
home_dir = os.path.expanduser("~")
log_file = os.path.join(home_dir, "laMEG_postinstallation.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Create a file handler for logging to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

# Create a console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

# Add both handlers to the root logger
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)


def install_spm():
    """Install the tested DANC SPM Python release."""
    logging.info(
        "Installing DANC SPM Python %s...",
        DANC_SPM_VERSION,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        clone_dir = os.path.join(temp_dir, "DANC_spm_python")

        try:
            logging.info(
                "Downloading DANC SPM Python %s...",
                DANC_SPM_VERSION,
            )

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    DANC_SPM_VERSION,
                    "--single-branch",
                    DANC_SPM_REPO,
                    clone_dir,
                ]
            )

            logging.info("Installing SPM package...")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-v",
                    clone_dir,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            logging.info(result.stdout)

            if result.stderr:
                logging.info(result.stderr)

            logging.info(
                "DANC SPM Python %s installed successfully.",
                DANC_SPM_VERSION,
            )

        except subprocess.CalledProcessError as err:
            logging.error(
                "Failed to install DANC SPM Python %s.",
                DANC_SPM_VERSION,
            )

            if getattr(err, "stdout", None):
                logging.error("stdout:\n%s", err.stdout)

            if getattr(err, "stderr", None):
                logging.error("stderr:\n%s", err.stderr)

            raise


def setup_jupyter_extensions():
    """
    Set up Jupyter notebook extensions.

    This method creates a script to install and enable the Jupyter `k3d`
    extension for the environment. The script will be executed when the
    environment is activated.
    """
    conda_env_path = os.path.dirname(os.path.dirname(sys.executable))
    activate_script_dir = os.path.join(conda_env_path, "etc", "conda", "activate.d")
    os.makedirs(activate_script_dir, exist_ok=True)

    activate_script_path = os.path.join(activate_script_dir, "jupyter_setup.sh")

    with open(activate_script_path, "w", encoding="utf-8") as out_file:
        out_file.write("#!/bin/bash\n\n")
        out_file.write("# Script to set up Jupyter extensions for the environment\n")

        # Marker file to prevent re-running the setup
        marker_file = os.path.join(conda_env_path, ".jupyter_setup_done")
        out_file.write(f'MARKER_FILE="{marker_file}"\n')
        out_file.write('if [ ! -f "$MARKER_FILE" ]; then\n')
        out_file.write("    echo 'Setting up Jupyter extensions...'\n")
        out_file.write("    if command -v jupyter &> /dev/null; then\n")
        out_file.write("        jupyter nbextension install --py --user k3d\n")
        out_file.write("        jupyter nbextension enable --py --user k3d\n")
        out_file.write("        echo 'Jupyter extensions setup completed.'\n")
        out_file.write('        touch "$MARKER_FILE"\n')
        out_file.write("    else\n")
        out_file.write(
            "        echo 'Jupyter is not installed. "
            "Please install Jupyter and try again.'\n"
        )
        out_file.write("    fi\n")
        out_file.write("fi\n")

    # Make the script executable
    os.chmod(activate_script_path, 0o755)
    logging.info("Jupyter setup script created and made executable.")


def run_postinstall():
    """Run all laMEG post-installation setup tasks."""
    logging.info("Running laMEG post-installation setup...")

    install_spm()
    setup_jupyter_extensions()

    logging.info("laMEG post-installation setup completed successfully.")

    # Detect if we're running inside a conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(
            f"Detected conda environment: '{conda_env}'.\n"
            "Before using laMEG, please deactivate and reactivate "
            "your environment\n"
            "so that environment variable changes take effect:\n\n"
            "    conda deactivate\n"
            f"    conda activate {conda_env}\n"
        )

    # ------------------------------------------------------------------
    # Create marker file so that __init__.py knows postinstall has run
    # ------------------------------------------------------------------
    marker_path = os.path.join(os.path.expanduser("~"), ".lameg_postinstall")

    try:
        with open(marker_path, "w", encoding="utf-8") as file:
            file.write("Post-installation completed successfully.\n")
    except OSError as err:
        print(
            "Warning: could not create postinstall marker file "
            f"({marker_path}): {err}"
        )


if __name__ == "__main__":
    run_postinstall()
