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

def check_dependencies():
    """Check if required build dependencies are installed."""
    missing_deps = {}
    found_deps = {}
    
    # Check for CMake
    try:
        cmake_result = subprocess.run(
            ["cmake", "--version"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        found_deps["CMake"] = cmake_result.stdout.strip().split('\n')[0] if cmake_result.stdout else "Found"
    except (subprocess.SubprocessError, FileNotFoundError):
        missing_deps["CMake"] = "https://cmake.org/download/"
    
    # Check for compilers based on platform
    compilers_found = False
    
    if platform.system() == "Windows":
        # Check multiple Windows compilers
        compiler_checks = [
            {"name": "MSVC", "cmd": ["cl"], "link": "https://visualstudio.microsoft.com/downloads/"},
            {"name": "GCC", "cmd": ["g++", "--version"], "link": "https://www.mingw-w64.org/downloads/"},
            {"name": "Clang", "cmd": ["clang", "--version"], "link": "https://releases.llvm.org/download.html"}
        ]
        
        for compiler in compiler_checks:
            try:
                result = subprocess.run(
                    compiler["cmd"], 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                found_deps[compiler["name"]] = result.stdout.strip().split('\n')[0] if result.stdout else "Found"
                compilers_found = True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Just note it was checked but not found
                pass
        
        if not compilers_found:
            missing_deps["C++ Compiler"] = {
                "MSVC": "https://visualstudio.microsoft.com/downloads/",
                "GCC (MinGW)": "https://www.mingw-w64.org/downloads/",
                "Clang": "https://releases.llvm.org/download.html"
            }
    else:
        # Unix-like systems (Linux, macOS)
        compiler_checks = [
            {"name": "GCC", "cmd": ["g++", "--version"], "link": "Install via your package manager (apt-get install g++)"},
            {"name": "Clang", "cmd": ["clang++", "--version"], "link": "Install via your package manager (apt-get install clang)"}
        ]
        
        for compiler in compiler_checks:
            try:
                result = subprocess.run(
                    compiler["cmd"], 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                found_deps[compiler["name"]] = result.stdout.strip().split('\n')[0] if result.stdout else "Found"
                compilers_found = True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Just note it was checked but not found
                pass
        
        if not compilers_found:
            missing_deps["C++ Compiler"] = {
                "GCC": "Install via your package manager (apt-get install g++)",
                "Clang": "Install via your package manager (apt-get install clang)"
            }
    
    # Check for Git if needed
    try:
        git_result = subprocess.run(
            ["git", "--version"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        found_deps["Git"] = git_result.stdout.strip() if git_result.stdout else "Found"
    except (subprocess.SubprocessError, FileNotFoundError):
        missing_deps["Git"] = "https://git-scm.com/downloads"
    
    return missing_deps, found_deps

def build_libraries(user_install: bool = False):
    """
    Function to compile the Shuriken library using CMake.

    :param user_install: If True, install for current user only
    """

    # Check dependencies first
    missing_deps, found_deps = check_dependencies()

    # Log what we found
    if found_deps:
        logger.info("Found build dependencies:")
        for dep, version in found_deps.items():
            logger.info(f"  - {dep}: {version}")
    
    if missing_deps:
        error_msg = ["Missing required build dependencies:"]
        
        for dep, info in missing_deps.items():
            if dep == "C++ Compiler":
                error_msg.append(f"\n- {dep}: No compatible C++ compiler found. Please install one of the following:")
                for compiler, link in info.items():
                    error_msg.append(f"  * {compiler}: {link}")
            else:
                error_msg.append(f"\n- {dep}: {info}")
        
        error_msg.append("\nInstallation cannot continue until these dependencies are resolved.")
        raise RuntimeError("\n".join(error_msg))

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
        logger.info("Checking build dependencies...")
        missing_deps, found_deps = check_dependencies()
        
        if missing_deps:
            self._show_missing_deps_error(missing_deps)
        
        logger.info("Building C extensions...")
        build_libraries(user_install=self.user_install)
        super().run()
    
    def _show_missing_deps_error(self, missing_deps):
        """Format and raise an error for missing dependencies"""
        error_msg = ["Required build dependencies are missing:"]
        
        for dep, info in missing_deps.items():
            if dep == "C++ Compiler":
                error_msg.append(f"\n- {dep}: No compatible C++ compiler found. Please install one of the following:")
                for compiler, link in info.items():
                    error_msg.append(f"  * {compiler}: {link}")
            else:
                error_msg.append(f"\n- {dep}: {info}")
        
        error_msg.append("\nInstallation cannot continue until these dependencies are resolved.")
        raise RuntimeError("\n".join(error_msg))


cmdclass = {
    "sdist": _sdist,
    "build_ext": CustomBuildExt,
    "bdist_egg": _bdist_egg,
    "install": CustomInstallCommand,
}

setup(
    name="ShurikenAnalyzer",
    version="0.0.6",
    author="Fare9",
    author_email="kunai.static.analysis@gmail.com",
    description="Shuriken-Analyzer: A library for Dalvik Analysis",
    url="https://github.com/Shuriken-Group/Shuriken-Analyzer/",
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
