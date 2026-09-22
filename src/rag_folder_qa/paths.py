"""Keep ingested files inside the folder the user pointed at."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DOC_SUFFIXES = {".md", ".txt"}
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".rag_store",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
}


class PathOutsideRootError(ValueError):
    """A path resolved outside the configured documents root."""


def ensure_within_root(root: Path, path: Path) -> Path:
    """Resolve ``path`` and require it to stay under ``root``.

    Relative paths are interpreted relative to ``root``. Symlinks are
    resolved, so a link inside the folder that points outside is rejected.
    A sibling directory that merely shares a string prefix (``docs`` vs
    ``docs-extra``) is not treated as inside.
    """

    root_resolved = root.resolve()
    candidate = path if path.is_absolute() else root_resolved / path
    resolved = candidate.resolve()
    if not _is_within(resolved, root_resolved):
        raise PathOutsideRootError(
            f"{path} resolves to {resolved}, which is outside {root_resolved}."
        )
    return resolved


def relative_source(root: Path, path: Path) -> str:
    """Return a posix path relative to ``root`` after the containment check."""

    resolved_root = root.resolve()
    resolved = ensure_within_root(resolved_root, path)
    return resolved.relative_to(resolved_root).as_posix()


@dataclass(frozen=True)
class ScanResult:
    files: tuple[Path, ...]
    skipped: tuple[str, ...]


def scan_documents(root: Path) -> ScanResult:
    """Collect ``.md`` and ``.txt`` files that resolve inside ``root``.

    Hidden files and hidden directories are skipped. Directory symlinks are
    not followed. A file symlink whose target leaves ``root`` is recorded in
    ``skipped`` and is not returned.
    """

    root_resolved = root.resolve()
    if not root_resolved.is_dir():
        raise NotADirectoryError(f"{root} is not a directory.")

    files: list[Path] = []
    skipped: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root_resolved, followlinks=False):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in DOC_SUFFIXES:
                continue
            try:
                safe = ensure_within_root(root_resolved, path)
            except PathOutsideRootError as exc:
                skipped.append(str(exc))
                continue
            if safe.is_file():
                files.append(safe)
    files.sort(key=lambda item: item.relative_to(root_resolved).as_posix())
    return ScanResult(tuple(files), tuple(skipped))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
