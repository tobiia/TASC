from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path.cwd()
VENV = PROJECT_ROOT / ".venv"
CACHE = PROJECT_ROOT / "cache"

if sys.platform == "win32":
    PYTHON = VENV / "Scripts" / "python.exe"
    PIP = VENV / "Scripts" / "pip.exe"
else:
    PYTHON = VENV / "bin" / "python"
    PIP = VENV / "bin" / "pip"

FILE_URLS = [
    "https://zenodo.org/records/20636728/files/semeval2020_ulscd_eng_c1_sentence-transformers_all-mpnet-base-v2_L11.npz?download=1",
    "https://zenodo.org/records/20636728/files/semeval2020_ulscd_eng_c1_targeted_words.npz?download=1",
    "https://zenodo.org/records/20636728/files/semeval2020_ulscd_eng_c2_sentence-transformers_all-mpnet-base-v2_L11.npz?download=1",
    "https://zenodo.org/records/20636728/files/semeval2020_ulscd_eng_c2_targeted_words.npz?download=1",
    "https://zenodo.org/records/20636728/files/top2vec_semeval2020_ulscd_eng.pkl?download=1",
]


def run_block(cmd, cwd=None):
    print("\n>", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def start_process(cmd, cwd=None):
    print("\n>", " ".join(map(str, cmd)))
    return subprocess.Popen(cmd, cwd=cwd or PROJECT_ROOT)


def download_file(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://zenodo.org/",
    }

    req = Request(url, headers=headers)
    response = urlopen(req, timeout=60)

    headers_resp = response.headers

    if "Content-Disposition" in headers_resp:
        content_disposition = headers_resp["Content-Disposition"]
        filename = content_disposition.split("filename=")[-1].strip('"')
    else:
        filename = url.split("/")[-1]

    with open(CACHE / filename, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    print(f"Downloaded {filename}")
    return filename


def download_all_files(urls):
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download_file, url) for url in urls]

        for f in as_completed(futures):
            f.result()  # raises errors properly


def main():
    print("*" * 60)
    print("TASC Installer")
    print("*" * 60)

    if sys.version_info < (3, 12):
        print(f"\nERROR: Python 3.12+ required. Current: {sys.version}")
        sys.exit(1)

    if not VENV.exists():
        print("\n[1/2] Creating virtual environment...")

        if sys.platform == "win32":
            run_block(["py", "-3.12", "-m", "venv", str(VENV)])
        else:
            run_block(["python3.12", "-m", "venv", str(VENV)])
    else:
        print("\n[1/2] Virtual environment already exists.")

    print("\n[2/2] Upgrading pip...")
    run_block([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])

    # non-block pip install
    print("\nInstalling dependencies (pip running in background)...")

    pip_proc = start_process(
        [
            PYTHON,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
            "lib/LSC",
            "lib/Top2Vec",
        ]
    )

    # parallel file download
    print("\nDownloading files...")

    download_all_files(FILE_URLS)

    # wait for pip AFTER downloads
    print("\nWaiting for pip to finish...")
    pip_code = pip_proc.wait()

    if pip_code != 0:
        raise RuntimeError(f"pip failed with exit code {pip_code}")

    # spaCy model install after pip
    print("\n[5/5] Installing spaCy model...")
    run_block([PYTHON, "-m", "spacy", "download", "en_core_web_sm"])

    # verification
    print("\nVerifying installation...")

    run_block(
        [
            PYTHON,
            "-c",
            (
                "import colorama, numpy, pandas, transformers, spacy, tqdm, torch, "
                "sklearn, scipy, gensim, umap, hdbscan, wordcloud, "
                "sentence_transformers, hnswlib, fastapi, lsc, top2vec;"
                "spacy.load('en_core_web_sm');"
                "print('Verification successful')"
            ),
        ]
    )

    print("\n" + "*" * 60)
    print("Installation complete. Run your app to start.")
    print("*" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)
