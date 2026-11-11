"""Extract only the six known files. Never execute the archive's preprocessing script."""

import argparse
import zipfile
from pathlib import Path

EXPECTED = {
    "games_joined.csv",
    "players_joined.csv",
    "rounds_normalized.csv",
    "quality_report.json",
    "build_impactai.py",
    "README.md",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        if set(archive.namelist()) != EXPECTED:
            raise ValueError("Expected exactly the six documented cleaned dataset files")
        for name in sorted(EXPECTED):
            content = archive.read(name)
            path = args.output / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Refusing to replace different source file: {name}")
            path.write_bytes(content)
    print("Verified extraction to", args.output)


if __name__ == "__main__":
    main()
