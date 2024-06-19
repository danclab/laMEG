import os
import site
import subprocess
import sys
import importlib
import zipfile


def install_package():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "."])


def extract_matlab_runtime(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Create target directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Extract files one at a time
        for file_info in zip_ref.infolist():
            extracted_file_path = os.path.join(extract_to, file_info.filename)

            # Extract the file
            zip_ref.extract(file_info, extract_to)

            # Check if the file needs executable permissions
            if file_info.filename.endswith('.sh') or file_info.filename.endswith(
                    'install') or 'bin' in file_info.filename:
                # Ensure the file is there and set executable permission
                if os.path.isfile(extracted_file_path):
                    os.chmod(extracted_file_path, os.stat(extracted_file_path).st_mode | 0o755)

    return extract_to


def install_matlab_runtime(install_dir):
    install_script = os.path.join(install_dir, 'install')
    destination_dir = os.path.join(install_dir, 'MATLAB_Runtime')

    # Ensure the install script has execute permissions
    if not os.path.exists(install_script):
        raise FileNotFoundError(f"The install script was not found in {install_dir}")

    # os.chmod(install_script,
    #          os.stat(install_script).st_mode | 0o111)  # Add execute permissions for user, group, and others

    # Run the installation command with the destination folder
    subprocess.check_call(
        [install_script, '-mode', 'silent', '-agreeToLicense', 'yes', '-destinationFolder', destination_dir])
    return destination_dir


def create_activate_script(matlab_runtime_path, conda_env_path):
    activate_script_dir = os.path.join(conda_env_path, "etc", "conda", "activate.d")
    os.makedirs(activate_script_dir, exist_ok=True)
    activate_script_path = os.path.join(activate_script_dir, "env_vars.sh")
    with open(activate_script_path, "w") as f:
        f.write(f'export MATLAB_RUNTIME_DIR="{matlab_runtime_path}"\n')
        f.write(
            'export LD_LIBRARY_PATH="${MATLAB_RUNTIME_DIR}/v96/runtime/glnxa64:${MATLAB_RUNTIME_DIR}/v96/bin/glnxa64:${MATLAB_RUNTIME_DIR}/v96/sys/os/glnxa64:$LD_LIBRARY_PATH"\n')
        f.write('export XAPPLRESDIR="${MATLAB_RUNTIME_DIR}/v96/X11/app-defaults"\n')


def create_deactivate_script(conda_env_path):
    deactivate_script_dir = os.path.join(conda_env_path, "etc", "conda", "deactivate.d")
    os.makedirs(deactivate_script_dir, exist_ok=True)
    deactivate_script_path = os.path.join(deactivate_script_dir, "env_vars.sh")
    with open(deactivate_script_path, "w") as f:
        f.write('unset MATLAB_RUNTIME_DIR\n')
        f.write('unset LD_LIBRARY_PATH\n')
        f.write('unset XAPPLRESDIR\n')


def get_installed_package_dir(package_name):
    """Check all known site-packages directories for the installed package."""
    site_packages = site.getsitepackages()
    for site_package in site_packages:
        potential_path = os.path.join(site_package, package_name)
        if os.path.isdir(potential_path):
            return potential_path
    raise Exception(f"Package {package_name} not found in site-packages directories: {site_packages}")



def main():
    # Install the package first
    install_package()

    # Locate the installed package directory after ensuring installation is complete
    # Force Python to recognize new packages
    importlib.invalidate_caches()

    # Attempt to locate the installed package directory
    package_dir = get_installed_package_dir('lameg')

    matlab_runtime_zip = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MATLAB_Runtime_R2019a_Update_9_glnxa64.zip')
    matlab_runtime_extract_dir = os.path.join(package_dir, 'matlab_runtime')

    conda_env_path = os.path.dirname(os.path.dirname(sys.executable))

    # Extract MATLAB Runtime
    extracted_path = extract_matlab_runtime(matlab_runtime_zip, matlab_runtime_extract_dir)

    # Install MATLAB Runtime
    matlab_runtime_path = install_matlab_runtime(extracted_path)

    # Set environment variables in Conda environment scripts
    create_activate_script(matlab_runtime_path, conda_env_path)
    create_deactivate_script(conda_env_path)



if __name__ == "__main__":
    main()
