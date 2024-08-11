"""
This module configures the installation setup for the laMEG package, facilitating its installation
and post-installation steps.
"""
import shutil
import site
import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstall(install):
    """
    Custom installation class to handle additional setup tasks for the laMEG package.

    This class extends the standard installation process to include:
    - Installing the SPM (Statistical Parametric Mapping) package.
    - Installing the MATLAB runtime.
    - Setting up necessary environment variables.
    - Downloading and extracting test data.
    - Setting up Jupyter extensions.
    """


    def run(self):
        """
        Executes the custom installation process.

        This method runs the standard installation, installs additional components like
        SPM and MATLAB runtime, sets environment variables, and sets up Jupyter extensions.
        """
        self.clone_and_install_spm()
        super().run()
        self.download_and_extract_test_data()
        self.setup_jupyter_extensions()


    def clone_and_install_spm(self):
        """
        Clones the SPM repository and installs it.

        This method clones the DANC_spm_python repository from GitHub, then installs it using pip.
        """
        repo_url = "https://github.com/danclab/DANC_spm_python.git"
        clone_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'DANC_spm_python')

        if not os.path.exists(clone_dir):
            print(f"Cloning SPM repository from {repo_url}...")
            subprocess.check_call(['git', 'clone', repo_url, clone_dir])
        else:
            print("SPM repository already exists, skipping cloning.")

        print("Installing SPM package...")
        try:
            # Use the pip associated with the current Python interpreter
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-v', clone_dir])
        except subprocess.CalledProcessError as err:
            print(f"Failed to install SPM package. Error: {err}")
            raise
        shutil.rmtree(clone_dir)


    def download_file(self, url, save_path):
        """
        Downloads a file from a given URL to the specified path.

        Parameters:
        url (str): The URL to download the file from.
        save_path (str): The local path to save the downloaded file.
        """
        if not os.path.exists(save_path):
            subprocess.check_call(['wget', '-c', url, '-O', save_path])


    def download_and_extract_test_data(self):
        """
        Downloads and extracts test data from a given URL.

        This method downloads a compressed file containing test data and extracts it into the base
        directory.
        """
        base_dir = os.path.abspath(os.path.dirname(__file__))
        test_data_zip = os.path.join(base_dir, 'test_data.tar.gz')
        test_data_download_url = 'https://osf.io/mgz9q/download'
        self.download_file(test_data_download_url, test_data_zip)
        if os.path.exists(test_data_zip):
            subprocess.check_call(['tar', '-xzf', test_data_zip, '-C', base_dir])
            os.remove(test_data_zip)


    def setup_jupyter_extensions(self):
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


    def get_installed_package_dir(self, package_name):
        """
        Finds the installation directory of a specified package.

        Parameters:
        package_name (str): The name of the package to locate.

        Returns:
        str: The path to the installed package.

        Raises:
        FileNotFoundError: If the package is not found in the site-packages directories.
        """
        site_packages = site.getsitepackages()
        for site_package in site_packages:
            potential_path = os.path.join(site_package, package_name)
            if os.path.isdir(potential_path):
                return potential_path
        raise FileNotFoundError(f"Package {package_name} not found in site-packages directories: "
                                f"{site_packages}")


# Read the long description from the README.md
with open('README.md', 'r', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='lameg',
    version='0.0.1',
    description='A toolbox for laminar inference with MEG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DANC lab',
    author_email='james.bonaiuto@isc.cnrs.fr',
    url='https://github.com/danclab/laMEG',
    packages=find_packages(include=['lameg', 'lameg.*']),
    include_package_data=True,
    install_requires=[
        "nibabel==3.2.2",
        "vtk==9.3.0",
        "numpy==1.19.5",
        "scipy==1.5.4",
        "matplotlib==3.3.4",
        "joblib==1.1.1",
        "setuptools==59.6.0",
        "elephant==0.10.0",
        "neo==0.9.0",
        "quantities==0.13.0",
        "h5py==3.1.0",
        "mne==1.3.1",
        "notebook==6.4.10",
        "widgetsnbextension==3.6.6",
        "ipywidgets==7.8.1",
        "jupyterlab-widgets==1.1.7",
        "k3d==2.14.5"
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix'
    ],
    cmdclass={
        'install': CustomInstall,
    },
)
