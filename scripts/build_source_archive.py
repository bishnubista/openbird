#!/usr/bin/env python3
"""Build the deterministic source archive used by the Homebrew formula."""

from __future__ import annotations

import argparse
import gzip
import subprocess
import tarfile
from pathlib import Path


# Authoritative exclusion list for the release archive. This is the only filter
# that affects the produced tarball (the script does not use `git archive`, so
# `.gitattributes export-ignore` rules would have no effect here).
EXCLUDED_PREFIXES = ("Formula/", "Casks/", ".github/")


def tracked_files(repo: Path) -> list[str]:
    # Tracked files only: a release archive must reflect the committed tree, not
    # whatever happens to be untracked in the working directory. Including
    # `--others` would leak stray/build files into a public download and make
    # the archive non-reproducible.
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo,
        text=False,
    )
    files = output.decode().split("\0")
    return sorted(
        path
        for path in files
        if path and not any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    )


def add_file(tar: tarfile.TarFile, repo: Path, relpath: str, prefix: str) -> None:
    source = repo / relpath
    info = tar.gettarinfo(str(source), arcname=f"{prefix}/{relpath}")
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if info.isfile():
        with source.open("rb") as handle:
            tar.addfile(info, handle)
    else:
        tar.addfile(info)


def build_archive(repo: Path, version: str, output: Path) -> None:
    prefix = f"openbird-{version}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        # filename="" suppresses the gzip FNAME field; without it, GzipFile
        # embeds the output file's basename in the header, making the archive
        # bytes depend on the output path rather than purely on content.
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for relpath in tracked_files(repo):
                    add_file(tar, repo, relpath, prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", default=Path.cwd(), type=Path)
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    build_archive(args.repo.resolve(), version, args.output.resolve())


if __name__ == "__main__":
    main()
