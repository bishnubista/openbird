"""Security regression: ``openbird ingest`` must not follow symlinks that escape
the selected root and ingest external content.

Mirrors the confirmed repro: a symlink placed inside the selected directory that
points at an external file containing ``OUTSIDE_SECRET`` previously got ingested,
storing a resolved out-of-root ``file://`` URL. The walk must refuse it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from openbird import cli


# --------------------------------------------------------------------------- #
# _collect_files — the containment boundary (no embedder needed)              #
# --------------------------------------------------------------------------- #


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Build a selected ``root`` dir and a sibling ``outside`` dir."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    return root, outside


def test_collect_files_refuses_symlink_to_external_file(tmp_path):
    root, outside = _make_tree(tmp_path)
    secret = outside / "secret.txt"
    secret.write_text("OUTSIDE_SECRET")
    (root / "link.txt").symlink_to(secret)
    (root / "ok.txt").write_text("INROOT_OK")

    files, escaped = cli._collect_files(root, glob="*", max_bytes=1_000_000)

    names = sorted(f.name for f in files)
    assert names == ["ok.txt"]
    assert escaped == 1
    # The external content must never appear among the collected paths.
    assert not any(p.resolve() == secret.resolve() for p in files)


def test_collect_files_does_not_recurse_into_symlinked_dir(tmp_path):
    root, outside = _make_tree(tmp_path)
    sub = outside / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text("OUTSIDE_DEEP")
    (root / "dlink").symlink_to(sub, target_is_directory=True)
    (root / "ok.txt").write_text("INROOT_OK")

    files, escaped = cli._collect_files(root, glob="*", max_bytes=1_000_000)

    names = sorted(f.name for f in files)
    assert names == ["ok.txt"]
    # The escaping symlinked-dir entry is counted; its contents never surface.
    assert escaped >= 1
    assert not any("deep.txt" == p.name for p in files)


def test_collect_files_keeps_legitimate_in_root_files(tmp_path):
    root, _ = _make_tree(tmp_path)
    (root / "a.txt").write_text("A")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("B")
    # An in-root symlink that points back inside the root is fine.
    (root / "selflink.txt").symlink_to(root / "a.txt")

    files, escaped = cli._collect_files(root, glob="*", max_bytes=1_000_000)

    names = sorted(f.name for f in files)
    assert names == ["a.txt", "b.txt", "selflink.txt"]
    assert escaped == 0


def test_collect_files_single_external_symlink_arg_refused(tmp_path):
    root, outside = _make_tree(tmp_path)
    secret = outside / "secret.txt"
    secret.write_text("OUTSIDE_SECRET")
    link = root / "link.txt"
    link.symlink_to(secret)

    files, escaped = cli._collect_files(link, glob="*", max_bytes=1_000_000)

    assert files == []
    assert escaped == 1


def test_collect_files_single_in_root_file_arg_works(tmp_path):
    root, _ = _make_tree(tmp_path)
    f = root / "doc.txt"
    f.write_text("hello")

    files, escaped = cli._collect_files(f, glob="*", max_bytes=1_000_000)

    assert [p.name for p in files] == ["doc.txt"]
    assert escaped == 0


# --------------------------------------------------------------------------- #
# Full CLI ingest path — external content never reaches the store            #
# --------------------------------------------------------------------------- #


class _FakeStore:
    """Records add_observation calls; never embeds or touches the network."""

    def __init__(self) -> None:
        self.observations: list[dict] = []

    def add_observation(self, text, **kwargs):
        self.observations.append({"text": text, **kwargs})

    def close(self) -> None:
        pass


def test_ingest_command_does_not_store_external_symlink_content(
    tmp_path, monkeypatch
):
    """End-to-end repro: the CLI must not read/store OUTSIDE_SECRET, and the
    legitimate in-root file must still be ingested with an in-root file:// URL."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("OUTSIDE_SECRET")
    (root / "link.txt").symlink_to(secret)
    ok = root / "ok.txt"
    ok.write_text("INROOT_OK")

    fake = _FakeStore()
    # Swap the real (embedding) store for an in-memory recorder so the test is
    # deterministic and offline.
    monkeypatch.setattr(cli, "_store", lambda *a, **k: fake)

    res = CliRunner().invoke(cli.app, ["ingest", str(root)])

    assert res.exit_code == 0, res.output
    stored_text = "\n".join(o["text"] for o in fake.observations)
    assert "OUTSIDE_SECRET" not in stored_text
    assert "INROOT_OK" in stored_text
    # Exactly one observation: the in-root file. Its URL stays inside the root.
    assert len(fake.observations) == 1
    url = fake.observations[0]["url"]
    assert url == ok.as_uri()
    # The stored URL must not point at the external secret.
    assert secret.name not in url
