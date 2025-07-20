import os
import platform
import subprocess
import json
import zipfile
import tarfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.install import install as _install
import logging
from typing import Optional, Dict, Any

# Try to import requests, but don't fail if it's not available during build
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
SETUP_PATH = Path(
    __file__
).resolve()  # /Shuriken-Analyzer/shuriken/bindings/Python/setup.py
ROOT_FOLDER = SETUP_PATH.parents[3]  # /Shuriken-Analyzer
BUILD_FOLDER = ROOT_FOLDER / "build"

# GitHub asset configuration
GITHUB_REPO = "Shuriken-Group/Shuriken-Analyzer"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

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


def detect_preferred_compiler() -> Optional[str]:
    """
    Detect which compiler is available on the system for Linux builds.
    
    :return: Preferred compiler name or None
    """
    try:
        # Check for GCC first (more common)
        result = subprocess.run(["gcc", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "gcc"
    except:
        pass
    
    try:
        # Check for Clang
        result = subprocess.run(["clang", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return "clang"
    except:
        pass
    
    return None


def get_platform_asset_patterns() -> list[str]:
    """
    Get the asset filename patterns for the current platform.
    Returns multiple patterns in order of preference.
    
    :return: List of platform-specific asset patterns to match
    """
    system = platform.system().lower()
    patterns = []
    
    if system == "windows":
        patterns.append("windows-msvc")
    elif system == "linux":
        # Detect preferred compiler
        preferred_compiler = detect_preferred_compiler()
        if preferred_compiler == "gcc":
            patterns.extend(["linux-gcc", "linux-clang"])  # Prefer GCC
        elif preferred_compiler == "clang":
            patterns.extend(["linux-clang", "linux-gcc"])  # Prefer Clang
        else:
            patterns.extend(["linux-gcc", "linux-clang"])  # Default order
    elif system == "darwin":
        # No macOS assets available based on provided links
        logger.warning("No macOS assets available - will need to build from source")
        return []
    
    return patterns


def check_github_assets() -> Optional[Dict[str, Any]]:
    """
    Check if downloadable assets are available on GitHub releases.
    
    :return: Asset info dict if found, None otherwise
    """
    if not HAS_REQUESTS:
        logger.info("Requests module not available, skipping asset download")
        return None
        
    try:
        logger.info("Checking for downloadable assets on GitHub...")
        
        # Make request to GitHub API
        response = requests.get(GITHUB_API_URL, timeout=10)
        response.raise_for_status()
        
        release_data = response.json()
        assets = release_data.get("assets", [])
        
        if not assets:
            logger.info("No assets found in latest release")
            return None
        
        # Get platform-specific patterns
        platform_patterns = get_platform_asset_patterns()
        if not platform_patterns:
            logger.info("No assets available for current platform")
            return None
        
        logger.info(f"Looking for assets matching patterns: {platform_patterns}")
        
        # Find matching asset (try patterns in order of preference)
        for pattern in platform_patterns:
            for asset in assets:
                asset_name = asset["name"].lower()
                if pattern in asset_name:
                    logger.info(f"Found matching asset: {asset['name']} (pattern: {pattern})")
                    return {
                        "name": asset["name"],
                        "download_url": asset["browser_download_url"],
                        "size": asset["size"],
                        "version": release_data.get("tag_name", "unknown"),
                        "pattern": pattern
                    }
        
        logger.info(f"No assets found matching any platform patterns: {platform_patterns}")
        available_assets = [asset["name"] for asset in assets]
        logger.info(f"Available assets: {available_assets}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to check GitHub assets: {e}")
        return None


def download_and_extract_asset(asset_info: Dict[str, Any], install_dir: Path) -> bool:
    """
    Download and extract the asset to the specified directory.
    
    :param asset_info: Asset information from GitHub API
    :param install_dir: Directory to install the assets
    :return: True if successful, False otherwise
    """
    if not HAS_REQUESTS:
        logger.warning("Requests module not available, cannot download assets")
        return False
        
    try:
        asset_name = asset_info["name"]
        download_url = asset_info["download_url"]
        asset_size = asset_info["size"]
        
        logger.info(f"Downloading {asset_name} ({asset_size} bytes)...")
        
        # Create install directory
        install_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the asset
        response = requests.get(download_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = install_dir / asset_name
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded {asset_name} successfully")
        
        # Extract the archive
        logger.info(f"Extracting {asset_name}...")
        
        if asset_name.endswith(('.zip', '.ZIP')):
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(install_dir)
        elif asset_name.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tar.xz')):
            with tarfile.open(temp_file, 'r:*') as tar_ref:
                tar_ref.extractall(install_dir)
        else:
            logger.warning(f"Unknown archive format for {asset_name}")
            return False
        
        # Remove the temporary archive file
        temp_file.unlink()
        
        logger.info(f"Successfully extracted {asset_name} to {install_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download/extract asset: {e}")
        return False


def verify_asset_installation(install_dir: Path) -> bool:
    """
    Verify that the downloaded Shuriken assets are properly installed.
    
    :param install_dir: Installation directory to check
    :return: True if installation appears valid, False otherwise
    """
    try:
        # Check for expected Shuriken directories and files
        required_paths = [
            "bin",  # Directory containing binaries
            "lib",  # Directory containing libraries
            "include",  # Header files
        ]
        
        # Check for specific Shuriken executables (adjust based on actual binaries)
        expected_binaries = [
            "shuriken",  # Main executable
            "shuriken-analyzer",  # Alternative name
            # Add other expected binary names here
        ]
        
        # Verify required directories exist
        for required in required_paths:
            required_path = install_dir / required
            if not required_path.exists():
                logger.warning(f"Required directory not found: {required_path}")
                return False
        
        # Check for at least one expected binary
        bin_dir = install_dir / "bin"
        found_binary = False
        
        for binary_name in expected_binaries:
            # Check for binary with and without .exe extension
            binary_paths = [
                bin_dir / binary_name,
                bin_dir / f"{binary_name}.exe"
            ]
            
            for binary_path in binary_paths:
                if binary_path.exists() and binary_path.is_file():
                    logger.info(f"Found expected binary: {binary_path}")
                    found_binary = True
                    break
            
            if found_binary:
                break
        
        if not found_binary:
            logger.warning(f"No expected binaries found in {bin_dir}")
            # List what's actually there for debugging
            if bin_dir.exists():
                actual_files = list(bin_dir.iterdir())
                logger.info(f"Files found in bin directory: {[f.name for f in actual_files]}")
            return False
        
        logger.info("Asset installation verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying asset installation: {e}")
        return False


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


def build_libraries(user_install: bool = False, force_build: bool = False):
    """
    Function to compile the Shuriken library using CMake, or download pre-built assets.

    :param user_install: If True, install for current user only
    :param force_build: If True, skip asset download and force building from source
    """
    
    # Try to download assets first (unless force_build is True)
    if not force_build:
        asset_info = check_github_assets()
        if asset_info:
            # Determine installation directory
            if user_install:
                if platform.system() in ("Darwin", "Linux"):
                    install_dir = Path.home() / ".local"
                elif platform.system() == "Windows":
                    install_dir = Path.home() / "AppData" / "Local" / "Shuriken"
            else:
                if platform.system() == "Windows":
                    install_dir = Path("C:/Program Files/Shuriken")
                else:
                    install_dir = Path("/usr/local")
            
            logger.info(f"Attempting to download pre-built assets to {install_dir}")
            
            if download_and_extract_asset(asset_info, install_dir):
                if verify_asset_installation(install_dir):
                    logger.info(f"Successfully installed pre-built assets (version {asset_info['version']})")
                    logger.info("Skipping build process - using downloaded assets")
                    return
                else:
                    logger.warning("Asset verification failed, falling back to building from source")
            else:
                logger.warning("Asset download failed, falling back to building from source")
        else:
            logger.info("No compatible pre-built assets found, building from source")
    else:
        logger.info("Force build enabled, skipping asset download")

    # Fallback to original build process
    logger.info("Building from source...")
    
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


# Convenience function for checking assets without building
def check_available_assets():
    """
    Check what assets are available for download without building.
    
    :return: Asset info if available, None otherwise
    """
    return check_github_assets()


class CustomInstallCommand(_install):
    user_options = _install.user_options + [
        ("user-install", None, "Install the package in user space"),
        ("force-build", None, "Force building from source (skip asset download)")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.user_install = False
        self.force_build = False

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        build_libraries(user_install=self.user_install, force_build=self.force_build)
        super().run()


class CustomBuildExt(_build_ext):
    user_options = _build_ext.user_options + [
        ("user-install", None, "Install the package in user space"),
        ("force-build", None, "Force building from source (skip asset download)")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.user_install = False
        self.force_build = False

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        logger.info("Checking build dependencies...")
        
        # If not forcing build, try assets first (skip dependency check)
        if not self.force_build:
            asset_info = check_github_assets()
            if asset_info:
                logger.info("Pre-built assets available, attempting download...")
                build_libraries(user_install=self.user_install, force_build=self.force_build)
                super().run()
                return
        
        # Check dependencies for source build
        missing_deps, found_deps = check_dependencies()
        
        if missing_deps:
            self._show_missing_deps_error(missing_deps)
        
        logger.info("Building C extensions...")
        build_libraries(user_install=self.user_install, force_build=self.force_build)
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
        "requests",  # Added for asset downloading
    ],
)
