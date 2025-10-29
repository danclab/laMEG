"""
laMEG Post-installation Script
------------------------------

This module performs all post-installation setup tasks for the laMEG package.
It installs the DANC SPM Python interface, configures Jupyter notebook
extensions (e.g., `k3d`), and creates a marker file in the user?s home
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
import shutil
import subprocess

# Set up logging to both the console and a log file in the user's home directory
import sys

home_dir = os.path.expanduser("~")  # Get the user's home directory
log_file = os.path.join(home_dir, 'laMEG_postinstallation.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Create a file handler for logging to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Create a console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

# Add both handlers to the root logger
logging.getLogger().addHandler(file_handler)
logging.getLogger().addHandler(console_handler)


def clone_and_install_spm():
    """
    Clones the SPM repository and installs it.
    """
    repo_url = "https://github.com/danclab/DANC_spm_python.git"
    clone_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'DANC_spm_python')

    if not os.path.exists(clone_dir):
        logging.info("Cloning SPM repository from %s...", repo_url)
        print(f"Cloning SPM repository from {repo_url}...")
        subprocess.check_call(['git', 'clone', repo_url, clone_dir])
    else:
        logging.info("SPM repository already exists, skipping cloning.")
        print("SPM repository already exists, skipping cloning.")

    logging.info("Installing SPM package...")
    print("Installing SPM package...")
    try:
        # Capture output from the pip install command
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-v', clone_dir],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                check=True)

        # Log the detailed output from pip install
        logging.info(result.stdout)
        logging.error(result.stderr)  # Errors go to the error log

        # Also print to console
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            logging.info("SPM package installed successfully.")
        else:
            logging.error("SPM installation failed with return code %d", result.returncode)
            raise subprocess.CalledProcessError(result.returncode, result.args)
    except subprocess.CalledProcessError as err:
        logging.error("Failed to install SPM package. Error: %s", err)
        print(f"Failed to install SPM package. Error: {err}")
        raise
    shutil.rmtree(clone_dir)


def setup_jupyter_extensions():
    """
    Sets up Jupyter notebook extensions.

    This method creates a script to install and enable the Jupyter `k3d` extension for the
    environment. The script will be executed when the environment is activated.
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
        out_file.write("if [ ! -f \"$MARKER_FILE\" ]; then\n")
        out_file.write("    echo 'Setting up Jupyter extensions...'\n")
        out_file.write("    if command -v jupyter &> /dev/null; then\n")
        out_file.write("        jupyter nbextension install --py --user k3d\n")
        out_file.write("        jupyter nbextension enable --py --user k3d\n")
        out_file.write("        echo 'Jupyter extensions setup completed.'\n")
        out_file.write("        touch \"$MARKER_FILE\"\n")
        out_file.write("    else\n")
        out_file.write("        echo 'Jupyter is not installed. Please install Jupyter and try "
                       "again.'\n")
        out_file.write("    fi\n")
        out_file.write("fi\n")

    # Make the script executable
    os.chmod(activate_script_path, 0o755)
    logging.info("Jupyter setup script created and made executable.")


def run_postinstall():
    """Run all laMEG post-installation setup tasks."""
    print("Running laMEG post-installation setup...")
    clone_and_install_spm()
    setup_jupyter_extensions()
    print("laMEG post-installation setup completed successfully.")

    # Detect if we're running inside a conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(
            f"Detected conda environment: '{conda_env}'.\n"
            "Before using laMEG, please deactivate and reactivate your environment\n"
            "so that environment variable changes take effect:\n\n"
            f"    conda deactivate\n"
            f"    conda activate {conda_env}\n"
        )

    # ----------------------------------------------------------------------
    # Create marker file so that __init__.py knows postinstall has been run
    # ----------------------------------------------------------------------
    marker_path = os.path.join(os.path.expanduser("~"), ".lameg_postinstall")
    try:
        with open(marker_path, "w", encoding="utf-8") as file:
            file.write("Post-installation completed successfully.\n")
    except OSError as err:
        print(f"Warning: could not create postinstall marker file ({marker_path}): {err}")

if __name__ == "__main__":
    run_postinstall()
