#!/usr/bin/env python3
"""Inspect staged Git blobs before publishing a source-only release.

Uses only the standard library and never prints matched credential values.
Checks the index, so ignored local training artifacts may remain on disk.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".yml", ".yaml", ".toml", ".ini"}
TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE", "NOTICE"}
GENERATED_ROOTS = {
    "data", "Dataset", "checkpoints", "pretrained_models", "vposer_v1_0",
    "third_party", "new_runs", ".new_runs", "results", "eval_results",
    "analysis", "logs", "outputs", ".local", ".ollama_models", ".ollama_runtime",
}
PATTERNS = {
    "possible credential": re.compile(
        rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
        rb"|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}"
        rb"|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    ),
    "private network endpoint": re.compile(
        rb"https?://(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"
    ),
    "machine-specific path": re.compile(rb"/root/(?:dev|miniconda3)/"),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> int:
    try:
        entries = git("ls-files", "--stage", "-z").split(b"\0")
    except subprocess.CalledProcessError:
        print("Initialize Git and stage the release files before running this check.")
        return 1
    issues = []
    total = 0
    count = 0
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        name = raw_path.decode()
        path = Path(name)
        count += 1
        if stage != "0" or mode not in {"100644", "100755"}:
            issues.append((name, "non-regular or conflicted index entry"))
            continue
        if (
            path.parts[0] in GENERATED_ROOTS
            or path.parts[0].startswith("runs")
            or "__pycache__" in path.parts
            or (path.suffix not in TEXT_SUFFIXES and path.name not in TEXT_NAMES)
        ):
            issues.append((name, "not an allowed source/documentation file"))
        size = int(git("cat-file", "-s", oid))
        total += size
        if size > MAX_BYTES:
            issues.append((name, "exceeds 5 MiB"))
            continue
        data = git("cat-file", "blob", oid)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append((name, "not UTF-8 text"))
        if b"\0" in data:
            issues.append((name, "binary content"))
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                issues.append((name, label))
    if not count:
        issues.append(("<index>", "no staged files"))
    for name, reason in issues:
        print(f"FAIL {name}: {reason}")
    print(f"Checked {count} staged files, {total / 1024 / 1024:.2f} MiB, {len(issues)} issue(s).")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
