from pathlib import Path


def create_tasc_directory_structure(root="tasc_data"):
    """
    Create the expected TASC directory structure.

    Structure:
    tasc_data/
    ├── corpus1/
    └── corpus2/
    """

    root_path = Path(root)

    directories = [
        root_path,
        root_path / "corpus1",
        root_path / "corpus2",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Directory structure created under '{root_path}'")


if __name__ == "__main__":
    create_tasc_directory_structure()
