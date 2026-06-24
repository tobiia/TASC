import shutil
from pathlib import Path

PROJECT_ROOT = Path.cwd()
OUTPUT_DIR = PROJECT_ROOT / ".dist"
ZIP_NAME = f"tasc_v1_0_0"

EXCLUDE = {
    "__pycache__",
    "xl-lexeme",
    "build",
    "egg-info",
    "sample",
    "cache",
    "corpora",
}


def exclude(path: Path) -> bool:
    parts = path.parts

    if any(part.startswith(".") for part in parts):
        return True

    for name in EXCLUDE:
        if any(name in part for part in parts):
            return True

    return False


def copy_filtered(src: Path, dst: Path):
    for item in src.rglob("*"):
        if exclude(item):
            continue

        relative = item.relative_to(src)
        target = dst / relative

        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def build():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    staging = OUTPUT_DIR / "package"
    staging.mkdir(parents=True, exist_ok=True)

    print("Copying files...")
    copy_filtered(PROJECT_ROOT, staging)

    # create cache + corpora directories
    (staging / "corpora" / "dataset" / "corpus1").mkdir(parents=True, exist_ok=True)
    (staging / "corpora" / "dataset" / "corpus2").mkdir(parents=True, exist_ok=True)
    (staging / "cache").mkdir(parents=True, exist_ok=True)

    zip_path = shutil.make_archive(
        base_name=str(OUTPUT_DIR / ZIP_NAME),
        format="zip",
        root_dir=staging,
    )

    print(f"\nCreated: {zip_path}")


if __name__ == "__main__":
    build()
