from pathlib import Path
import subprocess
import sys
from colorama import init, Fore

init(autoreset=True)

PROJECT_ROOT = Path.cwd()
VENV = PROJECT_ROOT / ".venv"

if sys.platform == "win32":
    PYTHON = VENV / "Scripts" / "python.exe"
    PIP = VENV / "Scripts" / "pip.exe"
else:
    PYTHON = VENV / "bin" / "python"
    PIP = VENV / "bin" / "pip"


def run(*cmd, cwd=None):
    print("\n>", " ".join(str(x) for x in cmd))
    subprocess.check_call(
        [str(x) for x in cmd],
        cwd=cwd or PROJECT_ROOT,
    )


def main():
    print(Fore.CYAN + "*" * 60)
    print(Fore.CYAN + "TASC Installer")
    print(Fore.CYAN + "*" * 60)

    if sys.version_info < (3, 12):
        print(Fore.RED + f"\nERROR: Python 3.12+ is required. Current: {sys.version}")
        sys.exit(1)

    if not VENV.exists():
        print(Fore.CYAN + "\n[1/5] Creating virtual environment...")

        if sys.platform == "win32":
            run("py", "-3.12", "-m", "venv", str(VENV))
        else:
            run("python3.12", "-m", "venv", str(VENV))
    else:
        print(Fore.CYAN + "\n[1/5] Virtual environment already exists.")

    print(Fore.CYAN + "\n[2/5] Upgrading pip...")
    run(PYTHON, "-m", "pip", "install", "--upgrade", "pip")

    print(Fore.CYAN + "\n[3/5] Installing dependencies...")

    print(Fore.BLUE + "\nInstalling LSC...")
    run(PIP, "install", "lib/LSC")

    print(Fore.BLUE + "\nInstalling Top2Vec...")
    run(PIP, "install", "lib/Top2Vec")

    print(Fore.BLUE + "\nInstalling TASC dependencies...")
    run(PIP, "install", "-r", "requirements.txt")

    print(Fore.CYAN + "\n[4/5] Installing spaCy model...")
    run(PYTHON, "-m", "spacy", "download", "en_core_web_sm")

    print(Fore.CYAN + "\n[5/5] Verifying installation...")
    run(
        PYTHON,
        "-c",
        (
            "import numpy;"
            "import pandas;"
            "import transformers;"
            "import spacy;"
            "import tqdm;"
            "import torch;"
            "import sklearn;"
            "import scipy;"
            "import gensim;"
            "import umap;"
            "import hdbscan;"
            "import wordcloud;"
            "import sentence_transformers;"
            "import hnswlib;"
            "import fastapi;"
            "import lsc;"
            "import top2vec;"
            "spacy.load('en_core_web_sm');"
            "print('Verification successful')",
        ),
    )

    print(Fore.CYAN + "\n" + "*" * 60)
    print(Fore.GREEN + "Installation complete. Double-click Run.bat to start.")
    print(Fore.CYAN + "*" * 60)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(Fore.RED + f"\nInstallation failed (exit code {e.returncode}).")
        sys.exit(e.returncode)
