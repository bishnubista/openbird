#!/usr/bin/env python3
"""Privacy-safe OpenBird beta rehearsal harness.

This script summarizes real-data readiness using route labels, counts, and
status codes only. It must never print captured text, window titles, URLs, raw
JSON payloads, or command stderr from content-bearing commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = Path("/Applications/OpenBird.app")
DEFAULT_TIMEOUT = 30.0


class ParseError(Exception):
    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


@dataclass
class Row:
    status: str
    name: str
    detail: str


def _run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    stdin_devnull: bool = True,
) -> subprocess.CompletedProcess[str] | TimeoutError:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL if stdin_devnull else None,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TimeoutError("timeout")


def _json_cmd(
    cli: list[str],
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any] | None, Row | None]:
    result = _run(cli + args, env=env, timeout=timeout)
    if isinstance(result, TimeoutError):
        return None, Row("BLOCKED", " ".join(args[:2]), "timeout")
    if result.returncode != 0:
        return None, Row("BLOCKED", " ".join(args[:2]), f"rc={result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return None, Row("BLOCKED", " ".join(args[:2]), "unparseable json")
    if not isinstance(payload, dict):
        return None, Row("BLOCKED", " ".join(args[:2]), "unparseable json")
    return payload, None


def _need(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ParseError(path)
        cur = cur[part]
    return cur


def _need_list(obj: dict[str, Any], path: str) -> list[Any]:
    value = _need(obj, path)
    if not isinstance(value, list):
        raise ParseError(path)
    return value


def _need_dict(obj: dict[str, Any], path: str) -> dict[str, Any]:
    value = _need(obj, path)
    if not isinstance(value, dict):
        raise ParseError(path)
    return value


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)) or value is None:
        return str(value)
    return "non_scalar"


def _row_from_parse(name: str, exc: ParseError) -> Row:
    return Row("BLOCKED", name, f"unparseable {exc.path}")


def _summarize_stats(cli: list[str], timeout: float) -> tuple[Row, int | None]:
    payload, error = _json_cmd(cli, ["data", "stats"], timeout=timeout)
    if error:
        return Row(error.status, "data stats", error.detail), None
    try:
        observations = int(_need(payload, "observations"))  # type: ignore[arg-type]
        detail = (
            f"observations={observations} chunks={_scalar(_need(payload, 'chunks'))} "
            f"vectors={_scalar(_need(payload, 'vectors'))} "
            f"day_memories={_scalar(_need(payload, 'day_memories'))} "
            f"encryption_enabled={_scalar(_need(payload, 'encryption_enabled'))}"
        )
        return Row("PASS", "data stats", detail), observations
    except ParseError as exc:
        return _row_from_parse("data stats", exc), None


def _summarize_integrity(cli: list[str], timeout: float) -> Row:
    result = _run(cli + ["data", "integrity"], timeout=timeout)
    if isinstance(result, TimeoutError):
        return Row("BLOCKED", "data integrity", "timeout")
    if result.returncode == 0 and "integrity: ok" in result.stdout:
        return Row("PASS", "data integrity", "ok")
    return Row("BLOCKED", "data integrity", f"rc={result.returncode}")


def _summarize_preflight(cli: list[str], timeout: float) -> Row:
    # preflight emits useful diagnostic JSON even when it exits non-zero.
    result = _run(cli + ["preflight", "--json", "--probe-embedding"], timeout=timeout)
    if isinstance(result, TimeoutError):
        return Row("BLOCKED", "preflight", "timeout")
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return Row("BLOCKED", "preflight", f"rc={result.returncode} unparseable json")
    if not isinstance(payload, dict):
        return Row("BLOCKED", "preflight", f"rc={result.returncode} unparseable json")
    try:
        sqlite = _need_dict(payload, "sqlite")
        encryption = _need_dict(payload, "encryption")
        cloud = _need_dict(payload, "cloud")
        privacy = _need_dict(payload, "privacy")
        macos = _need_dict(payload, "macos")
        ollama = _need_dict(payload, "ollama")
        embedding = _need_dict(payload, "embedding")
        completion = _need_dict(payload, "completion")
        runtime_ok = _need(payload, "runtime_ok")
        release_gate_ok = _need(payload, "release_gate_ok")
        sqlite_vec = _need(sqlite, "vec_available")
        sqlite_fts5 = _need(sqlite, "fts5_available")
        encryption_status = _need(encryption, "status")
        encryption_verified = _need(encryption, "verified")
        macos_all_passed = _need(macos, "all_passed")
        detail = (
            f"rc={result.returncode} runtime_ok={_scalar(runtime_ok)} "
            f"release_gate_ok={_scalar(release_gate_ok)} "
            f"sqlite_vec={_scalar(sqlite_vec)} sqlite_fts5={_scalar(sqlite_fts5)} "
            f"encryption_status={_scalar(encryption_status)} "
            f"encryption_backend={_scalar(_need(encryption, 'backend'))} "
            f"encryption_verified={_scalar(encryption_verified)} "
            f"cloud_active={_scalar(_need(cloud, 'active'))} "
            f"allowlist_count={len(_need_list(privacy, 'allowlist'))} "
            f"blocklist_count={len(_need_list(privacy, 'blocklist'))} "
            f"macos_all_passed={_scalar(macos_all_passed)} "
            f"missing_model_count={len(_need_list(ollama, 'missing_models'))} "
            f"embedding_probed={_scalar(_need(embedding, 'probed'))} "
            f"embedding_dim_ok={_scalar(_need(embedding, 'dim_ok'))} "
            f"embedding_probed_dim={_scalar(_need(embedding, 'probed_dim'))} "
            f"completion_probed={_scalar(_need(completion, 'probed'))} "
            f"completion_ok={_scalar(_need(completion, 'ok'))}"
        )
        ready = (
            result.returncode == 0
            and runtime_ok is True
            and release_gate_ok is True
            and sqlite_vec is True
            and sqlite_fts5 is True
            and encryption_status == "encrypted"
            and encryption_verified is True
            and macos_all_passed is True
            and _need(embedding, "probed") is True
            and _need(embedding, "dim_ok") is True
            and _need(completion, "probed") is True
            and _need(completion, "ok") is True
        )
        return Row("PASS" if ready else "BLOCKED", "preflight", detail)
    except ParseError as exc:
        return _row_from_parse("preflight", exc)


def _summarize_timeline(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli, ["timeline", "--day", str(day), "--json"], timeout=timeout
    )
    name = f"timeline day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        sessions = _need_list(payload, "sessions")
        detail = (
            f"session_count={len(sessions)} "
            f"total_observations={_scalar(_need(payload, 'total_observations'))} "
            f"distinct_apps={_scalar(_need(payload, 'distinct_apps'))} "
            f"active_seconds={_scalar(_need(payload, 'active_seconds'))}"
        )
        return Row("PASS", name, detail)
    except ParseError as exc:
        return _row_from_parse(name, exc)


def _summarize_briefing(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli, ["briefing", "--day", str(day), "--json"], timeout=timeout
    )
    name = f"briefing day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        sources = _need_list(payload, "sources")
        text = _need(payload, "text")
        if not isinstance(text, str):
            raise ParseError("text")
        detail = (
            f"reasoning_route={_scalar(_need(payload, 'reasoning_route'))} "
            f"sources_total={_scalar(_need(payload, 'sources_total'))} "
            f"source_count={len(sources)} text_chars={len(text)}"
        )
        return Row("PASS", name, detail)
    except ParseError as exc:
        return _row_from_parse(name, exc)


def _summarize_productivity(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli, ["productivity", "--day", str(day), "--json"], timeout=timeout
    )
    name = f"productivity day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        detail = (
            f"route={_scalar(_need(payload, 'route'))} "
            f"egress={_scalar(_need(payload, 'egress'))} "
            f"day_offset={_scalar(_need(payload, 'day_offset'))} "
            f"local_date={_scalar(_need(payload, 'local_date'))} "
            f"status={_scalar(_need(payload, 'productivity_status'))}"
        )
        return Row("PASS", name, detail)
    except ParseError as exc:
        return _row_from_parse(name, exc)


def _summarize_day_memory(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli, ["day-memory", "show", "--day", str(day), "--json"], timeout=timeout
    )
    name = f"day-memory day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        day_memory = _need_dict(payload, "day_memory")
        dm_payload = _need_dict(payload, "day_memory.payload")
        detail = (
            f"source_count={_scalar(_need(day_memory, 'source_count'))} "
            f"payload_day_offset={_scalar(_need(dm_payload, 'day_offset'))} "
            f"session_count={len(_need_list(dm_payload, 'sessions'))} "
            f"workstream_count={len(_need_list(dm_payload, 'workstreams'))} "
            f"open_loop_count={len(_need_list(dm_payload, 'open_loops'))}"
        )
        return Row("PASS", name, detail)
    except ParseError as exc:
        return _row_from_parse(name, exc)


def _summarize_deep_brain_status(cli: list[str], timeout: float) -> Row:
    payload, error = _json_cmd(cli, ["deep-brain", "status", "--json"], timeout=timeout)
    if error:
        return Row(error.status, "deep-brain status", error.detail)
    try:
        detail = (
            f"route={_scalar(_need(payload, 'route'))} "
            f"egress={_scalar(_need(payload, 'egress'))} "
            f"cloud_blocked_reason_count={len(_need_list(payload, 'cloud_blocked_reasons'))} "
            f"ask_blocked_reason_count={len(_need_list(payload, 'ask_blocked_reasons'))}"
        )
        return Row("PASS", "deep-brain status", detail)
    except ParseError as exc:
        return _row_from_parse("deep-brain status", exc)


def _preview_detail(payload: dict[str, Any]) -> str:
    exclusions = _need_dict(payload, "exclusions")
    return (
        f"route={_scalar(_need(payload, 'route'))} "
        f"packet_build_route={_scalar(_need(payload, 'packet_build_route'))} "
        f"egress={_scalar(_need(payload, 'egress'))} "
        f"cloud_ready={_scalar(_need(payload, 'cloud_ready'))} "
        f"day_offset={_scalar(_need(payload, 'day_offset'))} "
        f"sources_total={_scalar(_need(payload, 'sources_total'))} "
        f"selected_source_count={len(_need_list(payload, 'selected_sources'))} "
        f"blocked_reason_count={len(_need_list(payload, 'blocked_reasons'))} "
        f"input_observations={_scalar(_need(exclusions, 'input_observations'))} "
        f"kept_observations={_scalar(_need(exclusions, 'kept_observations'))} "
        f"excluded_observations={_scalar(_need(exclusions, 'excluded_observations'))} "
        f"excluded_by_keys={len(_need_dict(exclusions, 'excluded_by'))}"
    )


def _summarize_deep_brain_preview(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli, ["deep-brain", "preview", "--day", str(day), "--json"], timeout=timeout
    )
    name = f"deep-brain preview day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        return Row("PASS", name, _preview_detail(payload))
    except ParseError as exc:
        return _row_from_parse(name, exc)


def _summarize_deep_brain_exclusion(cli: list[str], day: int, timeout: float) -> Row:
    payload, error = _json_cmd(
        cli,
        ["deep-brain", "preview", "--day", str(day), "--exclude-source", "capture", "--json"],
        timeout=timeout,
    )
    name = f"deep-brain exclude-source day {day}"
    if error:
        return Row(error.status, name, error.detail)
    try:
        exclusions = _need_dict(payload, "exclusions")
        input_count = int(_need(exclusions, "input_observations"))
        kept = int(_need(exclusions, "kept_observations"))
        excluded = int(_need(exclusions, "excluded_observations"))
        detail = _preview_detail(payload)
        if input_count > 0 and kept != input_count - excluded:
            return Row("FAIL", name, detail)
        if input_count > 0 and excluded <= 0:
            return Row("BLOCKED", name, f"no capture-source rows excluded; {detail}")
        return Row("PASS", name, detail)
    except (ParseError, TypeError, ValueError) as exc:
        if isinstance(exc, ParseError):
            return _row_from_parse(name, exc)
        return Row("BLOCKED", name, "unparseable exclusions")


def _temp_stats(cli: list[str], env: dict[str, str], timeout: float) -> int | None:
    payload, _error = _json_cmd(cli, ["data", "stats"], env=env, timeout=timeout)
    if payload is None:
        return None
    try:
        return int(_need(payload, "observations"))
    except (ParseError, TypeError, ValueError):
        return None


def _real_stats(cli: list[str], timeout: float) -> int | None:
    payload, _error = _json_cmd(cli, ["data", "stats"], timeout=timeout)
    if payload is None:
        return None
    try:
        return int(_need(payload, "observations"))
    except (ParseError, TypeError, ValueError):
        return None


def _real_export_check(cli: list[str], timeout: float) -> Row:
    tempdir = tempfile.mkdtemp(prefix="openbird-beta-real-export-")
    export_path = Path(tempdir) / "real-export.jsonl"
    try:
        obs_before = _real_stats(cli, timeout)
        if obs_before is None:
            return Row("BLOCKED", "real export explicit", "stats unavailable")
        if obs_before == 0:
            return Row("BLOCKED", "real export explicit", "empty real store")

        # This briefly materializes real captured content in a 0700 temp dir.
        # The export file is checked by mode/count only and deleted in finally;
        # a SIGKILL could leave a same-user 0600 temp export behind.
        result = _run(
            cli + ["data", "export", "--output", str(export_path), "--yes"],
            timeout=timeout,
        )
        if isinstance(result, TimeoutError):
            return Row("BLOCKED", "real export explicit", "timeout")
        obs_after = _real_stats(cli, timeout)
        if obs_after is None:
            return Row("BLOCKED", "real export explicit", "stats unavailable")
        if obs_after < obs_before:
            return Row(
                "BLOCKED",
                "real export explicit",
                f"store changed non-monotonically stats_before={obs_before} stats_after={obs_after}",
            )
        if result.returncode != 0 or not export_path.exists():
            return Row(
                "FAIL",
                "real export explicit",
                f"rc={result.returncode} file_exists={int(export_path.exists())}",
            )
        try:
            mode = stat.S_IMODE(export_path.stat().st_mode)
            line_count = sum(1 for _ in export_path.open("r", encoding="utf-8"))
        except Exception:
            return Row("BLOCKED", "real export explicit", "read error")
        status = "PASS" if mode == 0o600 and obs_before <= line_count <= obs_after else "FAIL"
        return Row(
            status,
            "real export explicit",
            f"mode={mode:o} lines={line_count} stats_before={obs_before} stats_after={obs_after}",
        )
    finally:
        export_path.unlink(missing_ok=True)
        shutil.rmtree(tempdir, ignore_errors=True)


def _consent_checks(cli: list[str], timeout: float) -> list[Row]:
    rows: list[Row] = []
    tempdir = tempfile.mkdtemp(prefix="openbird-beta-rehearsal-")
    env = os.environ.copy()
    env["OPENBIRD_DATA_DIR"] = tempdir
    env["OPENBIRD_DISABLE_KEYRING"] = "1"
    env.pop("OPENBIRD_REQUIRE_ENCRYPTION", None)
    try:
        observations = _temp_stats(cli, env, timeout)
        if observations is None:
            return [Row("BLOCKED", "consent temp store", "stats unavailable")]

        no_yes = Path(tempdir) / "export-no-yes.jsonl"
        result = _run(cli + ["data", "export", "--output", str(no_yes)], env=env, timeout=timeout)
        if isinstance(result, TimeoutError):
            rows.append(Row("BLOCKED", "export consent", "timeout"))
        elif result.returncode != 1 or no_yes.exists():
            rows.append(
                Row(
                    "FAIL",
                    "export consent",
                    f"rc={result.returncode} file_exists={int(no_yes.exists())}",
                )
            )
        else:
            rows.append(Row("PASS", "export consent", "noninteractive_refused file_absent=1"))

        yes_path = Path(tempdir) / "export-yes.jsonl"
        try:
            result = _run(
                cli + ["data", "export", "--output", str(yes_path), "--yes"],
                env=env,
                timeout=timeout,
            )
            if isinstance(result, TimeoutError):
                rows.append(Row("BLOCKED", "export explicit", "timeout"))
            elif result.returncode != 0 or not yes_path.exists():
                rows.append(
                    Row(
                        "FAIL",
                        "export explicit",
                        f"rc={getattr(result, 'returncode', 'timeout')} file_exists={int(yes_path.exists())}",
                    )
                )
            else:
                mode = stat.S_IMODE(yes_path.stat().st_mode)
                line_count = sum(1 for _ in yes_path.open("r", encoding="utf-8"))
                status = "PASS" if mode == 0o600 and line_count == observations else "FAIL"
                rows.append(
                    Row(
                        status,
                        "export explicit",
                        f"mode={mode:o} lines={line_count} expected_lines={observations}",
                    )
                )
        finally:
            yes_path.unlink(missing_ok=True)

        for label, args in [
            ("prune consent", ["data", "prune", "--older-than", "36500d"]),
            ("purge consent", ["data", "purge", "--since", "9999-01-01"]),
        ]:
            before = _temp_stats(cli, env, timeout)
            result = _run(cli + args, env=env, timeout=timeout)
            after = _temp_stats(cli, env, timeout)
            if isinstance(result, TimeoutError):
                rows.append(Row("BLOCKED", label, "timeout"))
            elif before is None or after is None:
                rows.append(Row("BLOCKED", label, "stats unavailable"))
            elif result.returncode == 0 or before != after:
                rows.append(
                    Row(
                        "FAIL",
                        label,
                        f"rc={result.returncode} observations_before={before} observations_after={after}",
                    )
                )
            else:
                aborted = "Aborted" in (result.stdout + result.stderr)
                rows.append(
                    Row(
                        "PASS",
                        label,
                        f"rc={result.returncode} observations_unchanged={after} abort_signal={int(aborted)}",
                    )
                )
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
    return rows


def _release_status(app: Path, timeout: float) -> Row:
    env = os.environ.copy()
    env["OPENBIRD_RELEASE_STATUS_APP_BUNDLE"] = str(app)
    result = _run(
        [str(ROOT / "script" / "release_status.sh"), "--beta-rehearsal"],
        env=env,
        timeout=timeout,
    )
    if isinstance(result, TimeoutError):
        return Row("BLOCKED", "release status", "timeout")
    status = "PASS" if result.returncode == 0 else "BLOCKED"
    return Row(status, "release status", f"rc={result.returncode}")


def _ask_selftest(app: Path, timeout: float) -> Row:
    result = _run(
        [
            str(ROOT / "script" / "verify_ask_app.sh"),
            "What did I work on yesterday?",
            "--app",
            str(app),
        ],
        timeout=timeout,
    )
    if isinstance(result, TimeoutError):
        return Row("BLOCKED", "ask app self-test", "timeout")
    if result.returncode == 0:
        return Row("PASS", "ask app self-test", "grounded")
    if result.returncode == 1:
        return Row("FAIL", "ask app self-test", "ungrounded")
    return Row("BLOCKED", "ask app self-test", f"rc={result.returncode}")


def _print_rows(rows: list[Row]) -> None:
    for row in rows:
        print(f"{row.status:7} {row.name}: {row.detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run privacy-safe OpenBird beta rehearsal checks. Exit 0 means all "
            "checks passed, 1 means at least one product invariant failed, and "
            "2 means at least one readiness gate was blocked."
        )
    )
    parser.add_argument("--app", type=Path, default=DEFAULT_APP, help="OpenBird.app bundle to rehearse.")
    parser.add_argument("--day", type=int, default=1, help="Day offset for day-memory checks.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-command timeout.")
    parser.add_argument("--ask-timeout", type=float, default=45.0, help="Ask self-test timeout.")
    args = parser.parse_args(argv)

    app = args.app
    cli_path = app / "Contents" / "MacOS" / "openbird-cli"
    rows: list[Row] = []
    print("OpenBird beta rehearsal (content-safe)")
    print(f"app={app}")
    if not cli_path.exists() or not os.access(cli_path, os.X_OK):
        rows.append(Row("BLOCKED", "selected app CLI", "missing"))
        _print_rows(rows)
        return 2
    cli = [str(cli_path)]

    rows.append(_release_status(app, args.timeout))
    rows.append(_summarize_preflight(cli, args.timeout))
    stats_row, _observations = _summarize_stats(cli, args.timeout)
    rows.append(stats_row)
    rows.append(_summarize_integrity(cli, args.timeout))
    rows.append(_real_export_check(cli, args.timeout))
    rows.append(_summarize_timeline(cli, 0, args.timeout))
    rows.append(_summarize_timeline(cli, args.day, args.timeout))
    rows.append(_summarize_briefing(cli, args.day, args.timeout))
    rows.append(_summarize_productivity(cli, args.day, args.timeout))
    rows.append(_summarize_day_memory(cli, args.day, args.timeout))
    rows.append(_summarize_deep_brain_status(cli, args.timeout))
    rows.append(_summarize_deep_brain_preview(cli, args.day, args.timeout))
    rows.append(_summarize_deep_brain_exclusion(cli, args.day, args.timeout))
    rows.extend(_consent_checks(cli, args.timeout))
    rows.append(_ask_selftest(app, args.ask_timeout))

    _print_rows(rows)
    if any(row.status == "FAIL" for row in rows):
        return 1
    if any(row.status == "BLOCKED" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
