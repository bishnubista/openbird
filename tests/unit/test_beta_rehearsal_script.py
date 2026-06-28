from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_repo(
    tmp_path: Path,
    *,
    bad_briefing: bool = False,
    slow_stats: bool = False,
    no_capture_exclusion: bool = False,
    bad_exclusion_accounting: bool = False,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "script").mkdir(parents=True)
    shutil.copy(ROOT / "script" / "beta_rehearsal.py", repo / "script" / "beta_rehearsal.py")
    (repo / "script" / "beta_rehearsal.py").chmod(0o755)

    _write_executable(
        repo / "script" / "release_status.sh",
        """#!/usr/bin/env bash
set -euo pipefail
test "${1:-}" = "--beta-rehearsal"
printf 'release status metadata only\\n'
exit 0
""",
    )
    _write_executable(
        repo / "script" / "verify_ask_app.sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'VERDICT: PASS - grounded=1 citations=1 derived=0 sources=1\\n'
exit 0
""",
    )

    app = tmp_path / "Applications" / "OpenBird.app"
    cli = app / "Contents" / "MacOS" / "openbird-cli"
    export_log = tmp_path / "exports.log"
    _write_executable(
        cli,
        f"""#!/usr/bin/env bash
set -euo pipefail
bad_briefing="{int(bad_briefing)}"
slow_stats="{int(slow_stats)}"
no_capture_exclusion="{int(no_capture_exclusion)}"
bad_exclusion_accounting="{int(bad_exclusion_accounting)}"
case "${{1:-}}" in
  data)
    case "${{2:-}}" in
      stats)
        if [ "$slow_stats" = "1" ]; then
          sleep 2
        fi
        cat <<'JSON'
{{"observations":2,"blobs":1,"chunks":2,"vectors":2,"day_memories":1,"encryption_enabled":true}}
JSON
        ;;
      integrity)
        printf 'integrity: ok\\n'
        ;;
      export)
        out=""
        yes=0
        while [ "$#" -gt 0 ]; do
          case "$1" in
            --output) out="$2"; shift 2 ;;
            --yes) yes=1; shift ;;
            *) shift ;;
          esac
        done
        if [ "$yes" -eq 0 ]; then
          exit 1
        fi
        printf '%s\\n' "$out" >> "{export_log}"
        printf '{{"text":"SECRET_EXPORT_TEXT"}}\\n{{"text":"SECRET_EXPORT_TEXT_2"}}\\n' > "$out"
        chmod 600 "$out"
        ;;
      prune)
        printf 'Aborted!\\n' >&2
        exit 1
        ;;
      purge)
        printf 'Aborted!\\n' >&2
        exit 1
        ;;
      *) exit 2 ;;
    esac
    ;;
  timeline)
    cat <<'JSON'
{{"day_offset":1,"total_observations":2,"distinct_apps":1,"active_seconds":60.0,
 "sessions":[{{"window":"SECRET_WINDOW_TITLE","app":"com.example.Secret","count":2}}]}}
JSON
    ;;
  briefing)
    if [ "$bad_briefing" = "1" ]; then
      cat <<'JSON'
{{"reasoning_route":"local_deterministic","sources_total":1,
 "sources":[{{"snippet":"SECRET_SNIPPET","url":"https://secret.example"}}],
 "memory_context":{{"unsafe":"SECRET_CONTEXT"}}}}
JSON
    else
      cat <<'JSON'
{{"reasoning_route":"local_deterministic","sources_total":1,
 "sources":[{{"snippet":"SECRET_SNIPPET","url":"https://secret.example"}}],
 "text":"SECRET_BRIEFING_TEXT"}}
JSON
    fi
    ;;
  productivity)
    cat <<'JSON'
{{"route":"productivity.local_facts","egress":"none","day_offset":1,
 "local_date":"2026-06-27","productivity_status":"local_facts_only",
 "productivity":{{"facts":{{"unsafe":"SECRET_FACT"}}}}}}
JSON
    ;;
  day-memory)
    cat <<'JSON'
{{"built":true,"day_memory":{{"source_count":2,
 "payload":{{"day_offset":1,
   "sessions":[{{"cues":["SECRET_SESSION"]}}],
   "workstreams":[{{"label":"SECRET_WORKSTREAM"}}],
   "open_loops":[{{"title":"SECRET_LOOP"}}]}}}}}}
JSON
    ;;
  deep-brain)
    case "${{2:-}}" in
      status)
        cat <<'JSON'
{{"route":"deep_brain.status","egress":"none",
 "cloud_blocked_reasons":["off"],"ask_blocked_reasons":["off"]}}
JSON
        ;;
      preview)
        if printf '%s\\n' "$*" | grep -q -- "--exclude-source"; then
          if [ "$bad_exclusion_accounting" = "1" ]; then
            cat <<'JSON'
{{"route":"deep_brain.preview","packet_build_route":"deterministic_distillation",
 "egress":"none_preview","cloud_ready":false,"day_offset":1,"sources_total":1,
 "selected_sources":[],"blocked_reasons":["off"],
 "exclusions":{{"input_observations":2,"kept_observations":1,
   "excluded_observations":0,"excluded_by":{{}}}}}}
JSON
            exit 0
          fi
          if [ "$no_capture_exclusion" = "1" ]; then
            cat <<'JSON'
{{"route":"deep_brain.preview","packet_build_route":"deterministic_distillation",
 "egress":"none_preview","cloud_ready":false,"day_offset":1,"sources_total":1,
 "selected_sources":[{{"snippet":"SECRET_NON_CAPTURE"}}],"blocked_reasons":["off"],
 "exclusions":{{"input_observations":2,"kept_observations":2,
   "excluded_observations":0,"excluded_by":{{}}}}}}
JSON
            exit 0
          fi
          cat <<'JSON'
{{"route":"deep_brain.preview","packet_build_route":"deterministic_distillation",
 "egress":"none_preview","cloud_ready":false,"day_offset":1,"sources_total":0,
 "selected_sources":[],"blocked_reasons":["off"],
 "exclusions":{{"input_observations":2,"kept_observations":1,
   "excluded_observations":1,"excluded_by":{{"source":1}}}}}}
JSON
        else
          cat <<'JSON'
{{"route":"deep_brain.preview","packet_build_route":"deterministic_distillation",
 "egress":"none_preview","cloud_ready":false,"day_offset":1,"sources_total":1,
 "selected_sources":[{{"snippet":"SECRET_DEEP_SOURCE"}}],"blocked_reasons":["off"],
 "exclusions":{{"input_observations":2,"kept_observations":2,
   "excluded_observations":0,"excluded_by":{{}}}}}}
JSON
        fi
        ;;
      *) exit 2 ;;
    esac
    ;;
  *) exit 2 ;;
esac
""",
    )
    return repo, app, export_log


def _run_rehearsal(
    repo: Path,
    app: Path,
    *,
    timeout: str = "5",
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [str(repo / "script" / "beta_rehearsal.py"), "--app", str(app), "--timeout", timeout],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_beta_rehearsal_outputs_counts_without_content_and_cleans_export(tmp_path: Path) -> None:
    repo, app, export_log = _make_repo(tmp_path)

    result = _run_rehearsal(repo, app)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "PASS" in out
    assert "observations=2" in out
    assert "session_count=1" in out
    assert "text_chars=" in out
    assert "SECRET_" not in out
    assert "https://secret.example" not in out
    assert result.stderr == ""
    exported = [Path(line) for line in export_log.read_text(encoding="utf-8").splitlines()]
    assert exported
    assert all(not path.exists() for path in exported)


def test_beta_rehearsal_blocks_unparseable_without_leaking_sibling_content(tmp_path: Path) -> None:
    repo, app, _export_log = _make_repo(tmp_path, bad_briefing=True)

    result = _run_rehearsal(repo, app)

    assert result.returncode == 2
    assert "BLOCKED briefing day 1: unparseable text" in result.stdout
    assert "SECRET_" not in result.stdout
    assert "https://secret.example" not in result.stdout


def test_fake_export_fixture_writes_private_mode_before_harness_deletes(tmp_path: Path) -> None:
    repo, app, export_log = _make_repo(tmp_path)
    result = _run_rehearsal(repo, app)

    assert result.returncode == 0
    # The harness reports the mode it observed before deleting the file.
    assert "export explicit: mode=600 lines=2 expected_lines=2" in result.stdout
    for line in export_log.read_text(encoding="utf-8").splitlines():
        assert not Path(line).exists()


def test_beta_rehearsal_timeout_flag_blocks_slow_commands(tmp_path: Path) -> None:
    repo, app, _export_log = _make_repo(tmp_path, slow_stats=True)

    result = _run_rehearsal(repo, app, timeout="0.1")

    assert result.returncode == 2
    assert "BLOCKED data stats: timeout" in result.stdout


def test_beta_rehearsal_missing_cli_is_readiness_blocked(tmp_path: Path) -> None:
    repo, app, _export_log = _make_repo(tmp_path)
    (app / "Contents" / "MacOS" / "openbird-cli").unlink()

    result = _run_rehearsal(repo, app)

    assert result.returncode == 2
    assert "BLOCKED selected app CLI: missing" in result.stdout


def test_beta_rehearsal_no_matching_capture_exclusion_is_blocked_not_failed(tmp_path: Path) -> None:
    repo, app, _export_log = _make_repo(tmp_path, no_capture_exclusion=True)

    result = _run_rehearsal(repo, app)

    assert result.returncode == 2
    assert "BLOCKED deep-brain exclude-source day 1: no capture-source rows excluded" in result.stdout
    assert "FAIL" not in result.stdout
    assert "SECRET_" not in result.stdout


def test_beta_rehearsal_exclusion_accounting_mismatch_is_failed(tmp_path: Path) -> None:
    repo, app, _export_log = _make_repo(tmp_path, bad_exclusion_accounting=True)

    result = _run_rehearsal(repo, app)

    assert result.returncode == 1
    assert "FAIL    deep-brain exclude-source day 1:" in result.stdout
