"""Tests for the `openbird prompts` CLI sub-app (PR2)."""

from __future__ import annotations

from typer.testing import CliRunner

from openbird import cli
from openbird.config import reset_settings_cache

runner = CliRunner()


def _env(tmp_path):
    return {"OPENBIRD_DATA_DIR": str(tmp_path)}


def test_list_shows_default_source(tmp_path):
    reset_settings_cache()
    try:
        res = runner.invoke(cli.app, ["prompts", "list"], env=_env(tmp_path))
        assert res.exit_code == 0
        assert "rag" in res.stdout and "default" in res.stdout
    finally:
        reset_settings_cache()


def test_show_persona_and_full(tmp_path):
    reset_settings_cache()
    try:
        res = runner.invoke(cli.app, ["prompts", "show", "rag"], env=_env(tmp_path))
        assert res.exit_code == 0 and "ANSWERING RULES" in res.stdout
        full = runner.invoke(
            cli.app, ["prompts", "show", "rag", "--full"], env=_env(tmp_path)
        )
        assert full.exit_code == 0
        # --full includes the locked scaffold + the fence tokens.
        assert "SECURITY RULES" in full.stdout and "SECURITY REMINDER" in full.stdout
    finally:
        reset_settings_cache()


def test_show_unknown_key_exits_2(tmp_path):
    reset_settings_cache()
    try:
        res = runner.invoke(cli.app, ["prompts", "show", "nope"], env=_env(tmp_path))
        assert res.exit_code == 2
    finally:
        reset_settings_cache()


def test_edit_scaffolds_pure_persona_file(tmp_path):
    reset_settings_cache()
    try:
        # No $EDITOR -> command just scaffolds and prints the path.
        res = runner.invoke(
            cli.app, ["prompts", "edit", "rag"], env={**_env(tmp_path), "EDITOR": ""}
        )
        assert res.exit_code == 0
        path = tmp_path / "prompts" / "rag.txt"
        assert path.exists()
        body = path.read_text(encoding="utf-8")
        # The file is PURE persona: it must NOT contain the locked scaffold text.
        assert "ANSWERING RULES" in body
        assert "SECURITY RULES" not in body and "SECURITY REMINDER" not in body
        # The locked scaffold is shown in CLI output (not the file).
        assert "LOCKED" in res.stdout
    finally:
        reset_settings_cache()


def test_edit_preserves_existing_file(tmp_path):
    reset_settings_cache()
    try:
        pd = tmp_path / "prompts"
        pd.mkdir()
        (pd / "rag.txt").write_text("MY CUSTOM PERSONA", encoding="utf-8")
        res = runner.invoke(
            cli.app, ["prompts", "edit", "rag"], env={**_env(tmp_path), "EDITOR": ""}
        )
        assert res.exit_code == 0
        assert (pd / "rag.txt").read_text(encoding="utf-8") == "MY CUSTOM PERSONA"
    finally:
        reset_settings_cache()


def test_reset_deletes_override(tmp_path):
    reset_settings_cache()
    try:
        pd = tmp_path / "prompts"
        pd.mkdir()
        (pd / "rag.txt").write_text("X", encoding="utf-8")
        res = runner.invoke(
            cli.app, ["prompts", "reset", "rag", "--yes"], env=_env(tmp_path)
        )
        assert res.exit_code == 0 and not (pd / "rag.txt").exists()
    finally:
        reset_settings_cache()


def test_edit_propagates_editor_failure(tmp_path):
    reset_settings_cache()
    try:
        res = runner.invoke(
            cli.app,
            ["prompts", "edit", "rag"],
            env={**_env(tmp_path), "EDITOR": "/bin/false", "VISUAL": ""},
        )
        # A failing $EDITOR must make the command exit non-zero, not silently 0.
        assert res.exit_code != 0
    finally:
        reset_settings_cache()


def test_edit_refuses_symlink_override(tmp_path):
    reset_settings_cache()
    try:
        pd = tmp_path / "prompts"
        pd.mkdir()
        target = tmp_path / "elsewhere.txt"
        target.write_text("x", encoding="utf-8")
        (pd / "rag.txt").symlink_to(target)
        res = runner.invoke(
            cli.app, ["prompts", "edit", "rag"], env={**_env(tmp_path), "EDITOR": ""}
        )
        assert res.exit_code == 2
        # Did not write through the symlink.
        assert target.read_text(encoding="utf-8") == "x"
    finally:
        reset_settings_cache()


def test_reset_removes_broken_symlink(tmp_path):
    reset_settings_cache()
    try:
        pd = tmp_path / "prompts"
        pd.mkdir()
        (pd / "rag.txt").symlink_to(tmp_path / "does-not-exist")  # broken symlink
        res = runner.invoke(
            cli.app, ["prompts", "reset", "rag", "--yes"], env=_env(tmp_path)
        )
        assert res.exit_code == 0
        assert not (pd / "rag.txt").is_symlink()  # entry removed
    finally:
        reset_settings_cache()


def test_validate_clean_and_failing(tmp_path):
    reset_settings_cache()
    try:
        ok = runner.invoke(cli.app, ["prompts", "validate"], env=_env(tmp_path))
        assert ok.exit_code == 0
        # An oversized override must make validate exit non-zero.
        pd = tmp_path / "prompts"
        pd.mkdir()
        (pd / "rag.txt").write_bytes(b"x" * (64 * 1024 + 10))
        bad = runner.invoke(cli.app, ["prompts", "validate", "rag"], env=_env(tmp_path))
        assert bad.exit_code == 2 and "too-large" in bad.stdout
    finally:
        reset_settings_cache()
