import os
import subprocess
import sys
import argparse


def install_package():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "."])


def install_matlab_engine(matlab_path):
    matlab_engine_path = os.path.join(matlab_path, "extern", "engines", "python")
    if os.path.exists(matlab_engine_path):
        print("Installing MATLAB Python engine...")

        # Use the project's build directory for MATLAB engine installation
        project_build_dir = os.path.join(os.getcwd(), "build")

        # Construct the build and install command
        install_command = f'cd "{matlab_engine_path}" && python setup.py build --build-base="{project_build_dir}" install'

        os.system(install_command)
    else:
        print("Invalid MATLAB directory. Please install the MATLAB Python engine manually.")


def main():
    parser = argparse.ArgumentParser(description="Install the laMEG package and MATLAB Python engine.")
    parser.add_argument("-m", "--matlab_path", help="Path to the MATLAB installation directory.")
    args = parser.parse_args()

    install_package()

    if args.matlab_path:
        install_matlab_engine(args.matlab_path)
    else:
        matlab_path = input("Enter your MATLAB directory path (e.g., /usr/local/MATLAB/R2018a): ")
        install_matlab_engine(matlab_path)


if __name__ == "__main__":
    main()
