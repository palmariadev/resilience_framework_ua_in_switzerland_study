"""Install project dependencies from `requirements.txt`."""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def install_requirements(requirements_path: Path) -> None:
    """Install dependencies listed in the requirements file.

    Args:
        requirements_path: Path to the requirements file.
    """
    logger.info("Starting dependency installation.")
    if not requirements_path.exists():
        logger.error("Requirements file not found at %s.", requirements_path)
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")

    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    logger.info("Installing dependencies from %s.", requirements_path)

    try:
        subprocess.check_call(command)
        logger.info("Dependency installation finished successfully.")
    except subprocess.CalledProcessError:
        logger.exception("Dependency installation failed.")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    install_requirements(Path("requirements.txt"))
# smart_install.py
import subprocess
import sys

# --- Configuration ---
requirements_file = 'requirements.txt'

def install_requirements(req_file):
    """
    Installs all packages from a requirements.txt file using the correct pip.
    Pip will automatically skip any packages that are already installed.
    """
    try:
        print(f"✨ Reading packages from '{req_file}'...")
        
        # This is the robust command that uses the correct environment's pip.
        # It's the programmatic equivalent of: py -m pip install -r requirements.txt
        command = [sys.executable, '-m', 'pip', 'install', '-r', req_file]
        
        print("🚀 Starting installation... (Pip will skip packages that are already satisfied)")
        subprocess.check_call(command)
        
        print("\n✅ Environment is up to date.")

    except FileNotFoundError:
        print(f"❌ ERROR: The file '{req_file}' was not found. Please create it.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Pip failed to install packages. See error below:\n{e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    install_requirements(requirements_file)