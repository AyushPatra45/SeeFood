"""Download the public Kaggle archive. No dataset or credentials are committed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/dansbecker/hot-dog-not-hot-dog"


def extract_archive(archive, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (output / item.filename).resolve()
            if not target.is_relative_to(output):
                raise ValueError("Archive contains an unsafe path")
        bundle.extractall(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--archive", help="Use an already downloaded Kaggle ZIP")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else output.parent / "hotdog.zip"
    if not archive.exists():
        print("Downloading public Kaggle archive...", flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=180) as response, archive.open("wb") as f:
                shutil.copyfileobj(response, f)
        except Exception:
            archive.unlink(missing_ok=True)
            raise RuntimeError("Download failed. Download the ZIP from Kaggle and use --archive PATH.") from None
    if not zipfile.is_zipfile(archive):
        raise ValueError("Response is not a ZIP. Download manually from Kaggle, then use --archive PATH.")
    extract_archive(archive, output)
    metadata = {"source": URL, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (output.parent / "download.json").write_text(json.dumps(metadata, indent=2))
    print(f"Dataset extracted to {output}; archive SHA256: {metadata['sha256']}")


if __name__ == "__main__":
    main()
