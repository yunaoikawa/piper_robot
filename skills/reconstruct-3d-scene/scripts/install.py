#!/usr/bin/env python3
"""Install the repository skill as a non-copying symbolic link."""

from __future__ import annotations

import argparse
from pathlib import Path


def install(skill_dir: Path, skills_dir: Path) -> Path:
    source = skill_dir.resolve()
    destination_root = skills_dir.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.is_symlink() and destination.resolve() == source:
        return destination
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"{destination} already exists and is not this repository skill"
        )
    destination.symlink_to(source, target_is_directory=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument(
        "--skill-dir",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = parser.parse_args(argv)
    print(install(Path(args.skill_dir), Path(args.skills_dir)))


if __name__ == "__main__":
    main()
