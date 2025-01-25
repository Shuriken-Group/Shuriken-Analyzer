import os
import platform
import subprocess
from pathlib import Path
from contextlib import contextmanager
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.install import install as _install
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
SETUP_PATH = Path(
    __file__
).resolve()  # /Shuriken-Analyzer/shuriken/bindings/Python/setup.py
ROOT_FOLDER = SETUP_PATH.parents[3]  # /Shuriken-Analyzer
BUILD_FOLDER = ROOT_FOLDER / "build"

logger.info(f"Root folder: {ROOT_FOLDER}")
logger.info(f"Build folder: {BUILD_FOLDER}")


@contextmanager
def change_directory(path: Path):
    """Context manager to change directory and return to the original one."""
    current_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(current_dir)


def build_libraries(user_install: bool = False):
    """
    Function to compile the Shuriken library using CMake.

    :param user_install: If True, install for current user only
    """

    # Clear and recreate build directory to avoid cache problems
    if BUILD_FOLDER.exists():
        logger.info("Removing old build directory...")
        try:
            import shutil

            shutil.rmtree(BUILD_FOLDER)
        except Exception as e:
            logger.error(f"Error removing build directory: {e}")
            raise

    BUILD_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        with change_directory(BUILD_FOLDER):
            # Configure CMake with installation prefix if user install
            cmake_args = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]

            if user_install:
                if platform.system() in ("Darwin", "Linux"):
                    install_prefix = Path.home() / ".local"
                elif platform.system() == "Windows":
                    install_prefix = Path.home() / "AppData" / "Local"
                logger.info(f"User installation prefix: {install_prefix}")
            else:
                if platform.system() == "Windows":
                    install_prefix = Path("C:/Program Files/Shuriken")
                else:
                    install_prefix = Path("/usr/local")
                logger.info(f"System installation prefix: {install_prefix}")

            cmake_args.append(f"-DCMAKE_INSTALL_PREFIX={install_prefix}")

            logger.info("Configuring with CMake...")
            subprocess.check_call(cmake_args)

            logger.info("Building with CMake...")
            build_args = ["cmake", "--build", "."]
            if platform.system() == "Windows":
                build_args.extend(["--config", "Release"])
            else:
                build_args.append("-j")
            subprocess.check_call(build_args)

            logger.info("Installing with CMake...")
            install_cmd = ["cmake", "--install", "."]

            # Only use sudo for system installation
            if not user_install and platform.system() in ("Darwin", "Linux"):
                if os.path.exists("/usr/bin/sudo"):
                    install_cmd.insert(0, "sudo")

            subprocess.check_call(install_cmd)

    except subprocess.CalledProcessError as e:
        logger.error(f"CMake build failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


class CustomInstallCommand(_install):
    user_options = _install.user_options + [
        ("user-install", None, "Install the package in user space")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.user_install = False

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        build_libraries(user_install=self.user_install)
        super().run()


class CustomBuildExt(_build_ext):
    user_options = _build_ext.user_options + [
        ("user-install", None, "Install the package in user space")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.user_install = False

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        logger.info("Building C extensions...")
        build_libraries(user_install=self.user_install)
        super().run()


cmdclass = {
    "sdist": _sdist,
    "build_ext": CustomBuildExt,
    "bdist_egg": _bdist_egg,
    "install": CustomInstallCommand,
}

setup(
    name="ShurikenAnalyzer",
    version="0.0.5",
    author="Fare9",
    author_email="kunai.static.analysis@gmail.com",
    description="Shuriken-Analyzer: A library for Dalvik Analysis",
    url="https://github.com/Fare9/Shuriken-Analyzer",
    packages=find_packages(),
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Add your dependencies here
    ],
)
