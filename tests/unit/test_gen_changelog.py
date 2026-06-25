"""Unit tests for the release changelog generator (scripts/gen_changelog.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gen_changelog.py"
_spec = importlib.util.spec_from_file_location("gen_changelog", _MODULE_PATH)
assert _spec and _spec.loader
gen_changelog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_changelog)


def test_groups_by_conventional_type() -> None:
    out = gen_changelog.build_changelog(
        [
            "feat(app): clickable citations (#106)",
            "fix(capture): single-instance flock (#105)",
            "perf(search): cache embeddings (#200)",
            "ci: add encryption gate (#100)",
        ]
    )
    assert "### Features\n- clickable citations (#106)" in out
    assert "### Fixes\n- single-instance flock (#105)" in out
    assert "### Performance\n- cache embeddings (#200)" in out
    # ci collapses into Maintenance
    assert "### Maintenance\n- add encryption gate (#100)" in out


def test_skips_release_plumbing() -> None:
    out = gen_changelog.build_changelog(
        [
            "feat(app): real feature (#1)",
            "chore(release): bump version to 0.3.0 (#112)",
            "chore(homebrew): bump cask to 0.3.0 (#114)",
        ]
    )
    assert "real feature (#1)" in out
    assert "bump version" not in out
    assert "bump cask" not in out


def test_pr_number_optional() -> None:
    out = gen_changelog.build_changelog(["fix(core): no PR reference"])
    assert "- no PR reference\n" in out
    assert "(#" not in out


def test_bang_breaking_marker_is_tolerated() -> None:
    out = gen_changelog.build_changelog(["feat(api)!: drop legacy route (#9)"])
    assert "### Features\n- drop legacy route (#9)" in out


def test_non_conforming_subject_falls_through_to_other() -> None:
    out = gen_changelog.build_changelog(["Merge branch 'main' into feature"])
    assert "### Other\n- Merge branch 'main' into feature" in out


def test_empty_input_yields_empty_string() -> None:
    assert gen_changelog.build_changelog([]) == ""
