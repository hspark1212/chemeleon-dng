import tarfile
from pathlib import Path
import requests
from tqdm import tqdm

# Figshare download URL for the checkpoint tar.gz file
FIGSHARE_URL = "https://figshare.com/ndownloader/files/54966305"


def download_file(url: str, filepath: Path) -> None:
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        desc=f"Downloading {filepath.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_tar_gz(filepath: Path, extract_to: Path) -> None:
    """Extract tar.gz file to specified directory."""
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=extract_to)


def ensure_checkpoints_downloaded(ckpt_dir: str = "ckpts") -> None:
    """Ensure checkpoints are downloaded and extracted."""
    ckpt_path = Path(ckpt_dir)

    # Check if checkpoint directory exists and has files
    if ckpt_path.exists() and any(ckpt_path.glob("*.ckpt")):
        print("Checkpoints already exist.")
        return

    print("Checkpoints not found. Downloading...")

    # Create checkpoint directory
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Download tar.gz file
    tar_file = ckpt_path / "checkpoints.tar.gz"
    download_file(FIGSHARE_URL, tar_file)

    # Extract tar.gz file
    print("Extracting checkpoints...")
    extract_tar_gz(tar_file, ckpt_path.parent)

    # Clean up tar.gz file
    tar_file.unlink()

    print(f"Checkpoints downloaded and extracted to {ckpt_path}")


def get_checkpoint_path(task: str, default_paths: dict) -> str:
    """Get checkpoint path, downloading if necessary."""
    ensure_checkpoints_downloaded()

    ckpt_path = default_paths.get(task)
    if ckpt_path and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    return ckpt_path
