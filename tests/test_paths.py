"""Paths used for ingest and citations have to stay under the docs root."""

import pytest

from rag_folder_qa.paths import (
    PathOutsideRootError,
    ensure_within_root,
    relative_source,
    scan_documents,
)


def test_file_inside_root_is_allowed(tmp_path):
    root = tmp_path / "docs"
    nested = root / "team"
    nested.mkdir(parents=True)
    document = nested / "faq.md"
    document.write_text("hello\n", encoding="utf-8")
    resolved = ensure_within_root(root, document)
    assert resolved == document.resolve()
    assert relative_source(root, document) == "team/faq.md"


def test_parent_relative_path_is_rejected(tmp_path):
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("revenue\n", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        ensure_within_root(root, outside)
    escaped = (root / "nested" / ".." / ".." / "secret.txt")
    with pytest.raises(PathOutsideRootError):
        ensure_within_root(root, escaped)


def test_sibling_prefix_is_not_inside(tmp_path):
    root = tmp_path / "docs"
    sibling = tmp_path / "docs-extra"
    root.mkdir()
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("nope\n", encoding="utf-8")
    with pytest.raises(PathOutsideRootError):
        ensure_within_root(root, secret)


def test_symlink_outside_root_is_skipped(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "handbook.md").write_text("Ship day is Thursday.\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("The revenue was 999999.\n", encoding="utf-8")
    (root / "leak.txt").symlink_to(outside)
    scan = scan_documents(root)
    assert [path.name for path in scan.files] == ["handbook.md"]
    assert scan.skipped
    assert "leak.txt" in scan.skipped[0]


def test_scan_skips_hidden_and_non_docs(tmp_path):
    root = tmp_path / "docs"
    (root / "team").mkdir(parents=True)
    (root / "ok.md").write_text("visible\n", encoding="utf-8")
    (root / "team" / "note.txt").write_text("nested\n", encoding="utf-8")
    (root / "ignore.png").write_text("image\n", encoding="utf-8")
    (root / ".secret.md").write_text("hidden\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "secret.md").write_text("no\n", encoding="utf-8")
    scan = scan_documents(root)
    assert [path.relative_to(root).as_posix() for path in scan.files] == [
        "ok.md",
        "team/note.txt",
    ]
