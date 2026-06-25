"""Unit tests for the capture subsystem: redaction, adapters, daemon.

These tests use a FAKE capture helper (canned JSON, either as in-memory lines or
a tiny ``python -c`` subprocess emitter) so no real Accessibility access, signed
bundle, or Ollama is required. The ingest sink is a lightweight fake recording
``add_observation`` calls.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openbird.capture import adapters, redact, volatility
from openbird.capture.daemon import (
    CaptureDaemon,
    CaptureStats,
    CaptureSupervisorError,
    HelperExitError,
    HelperUnavailableError,
    default_helper_cmd,
    parse_event,
)
from openbird.config import Settings
from openbird.memory.store import MemoryStore
from openbird.types import Observation

from tests.unit.conftest import FakeProvider


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class FakeStore:
    """Records add_observation calls; returns a valid Observation."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_observation(
        self,
        text: str,
        *,
        app=None,
        window=None,
        url=None,
        session_id=None,
        source: str,
        ts=None,
    ) -> Observation:
        self.calls.append(
            {
                "text": text,
                "app": app,
                "window": window,
                "url": url,
                "session_id": session_id,
                "source": source,
                "ts": ts,
            }
        )
        return Observation(
            id=f"obs{len(self.calls)}",
            content_hash="h",
            ts=ts or 0.0,
            app=app,
            window=window,
            url=url,
            session_id=session_id,
            source=source,
        )


@pytest.fixture
def allow_settings(tmp_path) -> Settings:
    """Settings allowlisting Mail/Slack/Safari, default blocklist active."""
    return Settings(
        data_dir=tmp_path,
        allowlist=["com.apple.mail", "com.tinyspeck.slackmacgap", "com.apple.Safari"],
    )


def _line(**fields) -> str:
    return json.dumps(fields)


# ---------------------------------------------------------------------------
# redact.decide — allowlist-first
# ---------------------------------------------------------------------------


def test_decide_rejects_when_not_allowlisted(allow_settings):
    d = redact.decide(
        app="com.unknown.app", window="w", text="hello", settings=allow_settings
    )
    assert not d.capture
    assert d.reason == "not_allowlisted"


def test_decide_accepts_allowlisted(allow_settings):
    d = redact.decide(
        app="com.apple.mail", window="Inbox", text="hi", settings=allow_settings
    )
    assert d.capture
    assert d.reason == "allowlisted"


def test_decide_empty_allowlist_captures_nothing(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=[])
    d = redact.decide(app="com.apple.mail", window="w", text="hi", settings=s)
    assert not d.capture
    assert d.reason == "not_allowlisted"


def test_decide_blocklist_subtracts_even_if_allowlisted(tmp_path):
    s = Settings(
        data_dir=tmp_path,
        allowlist=["com.microsoft.VSCode"],
        blocklist=["com.microsoft.VSCode"],
    )
    d = redact.decide(app="com.microsoft.VSCode", window="w", text="code", settings=s)
    assert not d.capture
    assert d.reason == "blocklisted"


def test_decide_dangerous_app_backstop(tmp_path):
    # Even if the user mistakenly allowlists a password manager and clears the
    # blocklist, the hardcoded dangerous-category backstop blocks it.
    s = Settings(
        data_dir=tmp_path,
        allowlist=["com.1password.1password"],
        blocklist=[],
    )
    d = redact.decide(
        app="com.1password.1password", window="Vault", text="secret", settings=s
    )
    assert not d.capture
    assert d.reason == "dangerous_app"


def test_decide_incognito_flag(allow_settings):
    d = redact.decide(
        app="com.apple.Safari",
        window="Some Page",
        text="content",
        incognito=True,
        settings=allow_settings,
    )
    assert not d.capture
    assert d.reason == "incognito"


def test_decide_incognito_title_heuristic(allow_settings):
    d = redact.decide(
        app="com.apple.Safari",
        window="Reddit (Private Browsing)",
        text="content",
        settings=allow_settings,
    )
    assert not d.capture
    assert d.reason == "incognito"


def test_decide_no_text(allow_settings):
    d = redact.decide(app="com.apple.mail", window="w", text="   ", settings=allow_settings)
    assert not d.capture
    assert d.reason == "no_text"


# ---------------------------------------------------------------------------
# redact.scrub — secret masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret, rule",
    [
        ("sk-abcdefghijklmnop1234567890", "token_prefixed"),
        ("ghp_0123456789abcdefghijABCDEFGHIJ", "token_prefixed"),
        ("AKIAIOSFODNN7EXAMPLE", "token_prefixed"),
        ("password: hunter2longer", "secret_assignment"),
        ("api_key = abcdef1234567890", "secret_assignment"),
        ("4111 1111 1111 1111", "card_number"),
        ("123-45-6789", "ssn"),
    ],
)
def test_scrub_masks_secrets(secret, rule):
    text = f"prefix text {secret} suffix text"
    scrubbed, matched = redact.scrub(text)
    assert rule in matched
    # The raw secret payload should not survive verbatim.
    assert secret not in scrubbed
    assert "REDACTED" in scrubbed


def test_scrub_leaves_normal_text():
    text = "The meeting is at 3pm in room 204 about the roadmap."
    scrubbed, matched = redact.scrub(text)
    assert scrubbed == text
    assert matched == ()


def test_scrub_jwt_and_pem():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQabcdef"
    scrubbed, matched = redact.scrub(f"token={jwt}")
    assert "jwt" in matched
    assert jwt not in scrubbed


# ---------------------------------------------------------------------------
# credit-card (PAN) redaction: Luhn-validated, group-anchored.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pan",
    [
        "4111 1111 1111 1111",        # valid 16-digit Visa (separated)
        "4111111111111111",           # valid 16-digit Visa (continuous)
        "4000000000006",              # valid 13-digit
        "378282246310005",            # valid 15-digit Amex
        "6045632000000000120",        # valid 19-digit (Maestro range)
        "6045 6320 0000 0000 120",    # valid 19-digit, separated
    ],
)
def test_scrub_masks_valid_pans(pan):
    scrubbed, matched = redact.scrub(f"my card is {pan} thanks")
    assert "card_number" in matched
    assert "[REDACTED:card]" in scrubbed
    # No long run of the original PAN digits survives.
    assert pan.replace(" ", "") not in scrubbed.replace(" ", "")


@pytest.mark.parametrize(
    "not_a_pan",
    [
        "1234567890123456",            # 16-digit run that FAILS Luhn (phone-like)
        "1234 5678 9012 3456",         # same, separated, fails Luhn
        "123456789012345678",          # 18-digit legit order/account ID (fails Luhn)
        "12345678901234567890",        # 20-digit run: not a card length at all
        "order 30000000000004567 ref", # long ID; never inner-scanned
    ],
)
def test_scrub_preserves_non_pans(not_a_pan):
    text = f"reference {not_a_pan} end"
    scrubbed, matched = redact.scrub(text)
    assert "card_number" not in matched
    assert scrubbed == text


def test_scrub_card_plus_cvv_masks_only_pan():
    # '4111 1111 1111 1111 999' is 19 digits and fails Luhn as a whole, but the
    # first four groups (16 digits) are a valid PAN. We mask the PAN groups only
    # and leave the trailing CVV-like group intact (group-boundary masking).
    scrubbed, matched = redact.scrub("pay 4111 1111 1111 1111 999 now")
    assert "card_number" in matched
    assert "[REDACTED:card]" in scrubbed
    assert "4111" not in scrubbed
    assert scrubbed.endswith("999 now")


def test_scrub_continuous_run_with_embedded_valid_window_not_masked():
    # A continuous 18-digit ID can contain a Luhn-valid 16-digit inner window;
    # we NEVER scan arbitrary inner windows on a continuous run, so it survives.
    embedded = "4111111111111111"  # valid 16-digit window
    text = f"id 99{embedded} stop"  # 18 continuous digits, whole fails Luhn
    scrubbed, matched = redact.scrub(text)
    assert "card_number" not in matched
    assert scrubbed == text


def test_scrub_separated_over_19_digits_masks_valid_prefix():
    # Regression note: separated runs whose TOTAL exceeds 19 digits — the longest VALID
    # 13-19 digit whole-group prefix is masked; trailing groups are preserved.
    scrubbed, matched = redact.scrub("4111 1111 1111 1111 1234 5678")
    assert "card_number" in matched
    assert scrubbed.endswith("1234 5678")
    assert "4111" not in scrubbed


def test_scrub_card_not_torn_from_alnum_token():
    # A PAN-shaped digit run embedded in an alnum identifier is bounded out by
    # the ASCII token lookarounds and not masked mid-token.
    text = "abc4111111111111111xyz"
    scrubbed, matched = redact.scrub(text)
    assert "card_number" not in matched
    assert scrubbed == text


def test_scrub_card_ignores_non_ascii_digits():
    # Fullwidth/Unicode decimal digits match Python's \d and str.isdigit(), but
    # the Luhn helper assumes ASCII. The candidate regex uses re.ASCII so these
    # are never treated as card candidates (no false [REDACTED:card]).
    text = "１２３４ ５６７８ ９０１２ ３４５６"  # fullwidth digits
    scrubbed, matched = redact.scrub(text)
    assert "card_number" not in matched
    assert scrubbed == text


# ---------------------------------------------------------------------------
# token/JWT boundaries must not rely on Unicode-aware \b.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    ["keyσ", "テスト", "secret中文"],  # Greek sigma, CJK
)
def test_scrub_token_abutting_non_ascii_redacts(prefix):
    key = "sk-abcdefghijklmnop1234567890"
    scrubbed, matched = redact.scrub(f"{prefix}{key} tail")
    assert "token_prefixed" in matched
    assert key not in scrubbed


def test_scrub_token_after_underscore_redacts():
    key = "sk-abcdefghijklmnop1234567890"
    scrubbed, matched = redact.scrub(f"value_{key}")
    assert "token_prefixed" in matched
    assert key not in scrubbed


@pytest.mark.parametrize(
    "wrapped",
    [
        "-sk-abcdefghijklmnop1234567890",          # leading dash boundary
        "sk-abcdefghijklmnop1234567890-",          # trailing dash boundary
        "ghp_0123456789abcdefghijABCDEFGHIJ-",     # trailing dash, no-dash body
        "(ghp_0123456789abcdefghijABCDEFGHIJ)",    # punctuation boundaries
        "AKIAIOSFODNN7EXAMPLE-",                   # trailing dash, AWS key
        "x-sk-abcdefghijklmnop1234567890-y",       # dash-delimited within text
    ],
)
def test_scrub_token_dash_punctuated_redacts(wrapped):
    # Regression: the ASCII lookarounds must treat ``-`` (and ``_``) as token
    # boundaries, so a dash-prefixed/suffixed key is still redacted (the old
    # ``\b`` caught these; a lookahead that rejected ``-`` would leak them).
    scrubbed, matched = redact.scrub(f"see {wrapped} end")
    assert "token_prefixed" in matched
    assert "[REDACTED:token]" in scrubbed


def test_scrub_jwt_dash_suffixed_redacts():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQabcdef"
    scrubbed, matched = redact.scrub(f"-{jwt}-")
    assert "jwt" in matched
    assert jwt not in scrubbed


def test_scrub_jwt_abutting_non_ascii_redacts():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQabcdef"
    scrubbed, matched = redact.scrub(f"令牌{jwt}結尾")
    assert "jwt" in matched
    assert jwt not in scrubbed


def test_scrub_token_glued_to_ascii_identifier_documented_leak():
    # DOCUMENTED limitation: a key glued behind an ASCII alnum char (e.g.
    # 'xsk-...') is treated as part of an identifier token and is NOT redacted.
    # The leading boundary class includes [A-Za-z0-9] so this is by design; we
    # assert it to make the behavior explicit (changing it risks tearing apart
    # legitimate identifiers/filenames).
    key = "sk-abcdefghijklmnop1234567890"
    scrubbed, matched = redact.scrub(f"x{key}")
    assert "token_prefixed" not in matched
    assert scrubbed == f"x{key}"


def test_apply_rejected_returns_no_text(allow_settings):
    decision, text = redact.apply(
        app="com.unknown", window="w", text="secret stuff", settings=allow_settings
    )
    assert not decision.capture
    assert text is None


def test_apply_accepted_scrubs(allow_settings):
    decision, text = redact.apply(
        app="com.apple.mail",
        window="Inbox",
        text="my token sk-abcdefghijklmnop1234567890 here",
        settings=allow_settings,
    )
    assert decision.capture
    assert text is not None
    assert "sk-abcdefghijklmnop" not in text
    assert "token_prefixed" in decision.matched_rules


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def test_compatibility_matrix_shape():
    assert "com.apple.Safari" in adapters.COMPATIBILITY_MATRIX
    for bundle_id, profile in adapters.COMPATIBILITY_MATRIX.items():
        assert profile.bundle_id == bundle_id
        assert profile.coverage in {"full", "partial", "degraded"}
        assert isinstance(profile.fields, tuple)


def test_coverage_for_known_and_unknown():
    assert adapters.coverage_for("com.apple.mail") == "full"
    assert adapters.coverage_for("us.zoom.xos") == "degraded"
    assert adapters.coverage_for("com.nonexistent.app") == "unknown"
    assert adapters.coverage_for(None) == "unknown"


def test_normalize_collapses_whitespace_and_blanks():
    raw = "Hello    world\n\n\n  Foo   bar  \n"
    out = adapters.normalize_for_app(raw, "com.apple.mail")
    assert out == "Hello world\nFoo bar"


def test_normalize_drops_repeated_lines_and_chrome():
    raw = "Search\nReal content\nReal content\nSettings\nMore content"
    out = adapters.normalize_for_app(raw, "com.tinyspeck.slackmacgap")
    lines = out.splitlines()
    assert "Real content" in lines
    assert lines.count("Real content") == 1  # immediate repeat collapsed
    assert "Search" not in lines  # generic chrome
    assert "Settings" not in lines


def test_normalize_app_specific_chrome():
    raw = "Threads\nThe actual message body\nHuddles"
    out = adapters.normalize_for_app(raw, "com.tinyspeck.slackmacgap")
    assert out == "The actual message body"


# ---------------------------------------------------------------------------
# parse_event
# ---------------------------------------------------------------------------


def test_parse_event_valid():
    e = parse_event(_line(app="a", window="w", url="u", text="t", ts=123.5))
    assert e == {
        "app": "a",
        "window": "w",
        "url": "u",
        "text": "t",
        "ts": 123.5,
        "incognito": False,
    }


def test_parse_event_blank_and_malformed():
    assert parse_event("") is None
    assert parse_event("   ") is None
    assert parse_event("{not json") is None
    assert parse_event("[1,2,3]") is None  # not an object


def test_parse_event_bad_ts_becomes_none():
    e = parse_event(_line(app="a", text="t", ts="not-a-number"))
    assert e is not None
    assert e["ts"] is None


# ---------------------------------------------------------------------------
# CaptureDaemon.run_lines — end-to-end with fake helper output
# ---------------------------------------------------------------------------


def test_daemon_ingests_allowlisted_event(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="Quarterly report ready", ts=10.0),
    ]
    stats = daemon.run_lines(lines)
    assert stats.received == 1
    assert stats.ingested == 1
    assert stats.rejected == 0
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["app"] == "com.apple.mail"
    assert call["source"] == "capture"
    assert call["ts"] == 10.0
    assert "Quarterly report" in call["text"]


def test_daemon_coalesces_unchanged_recent_capture(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings, duplicate_window=60.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=12.0),
    ]

    stats = daemon.run_lines(lines)

    assert stats.received == 2
    assert stats.ingested == 1
    assert stats.coalesced == 1
    assert stats.rejected == 0
    assert len(store.calls) == 1


def test_daemon_keeps_heartbeat_for_unchanged_capture(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings, duplicate_window=60.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=69.0),
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=70.0),
    ]

    stats = daemon.run_lines(lines)

    assert stats.ingested == 2
    assert stats.coalesced == 1
    assert [call["ts"] for call in store.calls] == [10.0, 70.0]


def test_daemon_does_not_coalesce_backward_timestamps(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings, duplicate_window=60.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=9.0),
    ]

    stats = daemon.run_lines(lines)

    assert stats.ingested == 2
    assert stats.coalesced == 0


def test_daemon_ingests_changed_capture_immediately(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings, duplicate_window=60.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="first report", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="updated report", ts=12.0),
        _line(app="com.apple.mail", window="Thread", text="updated report", ts=14.0),
    ]

    stats = daemon.run_lines(lines)

    assert stats.ingested == 3
    assert stats.coalesced == 0
    assert [call["text"] for call in store.calls] == [
        "first report",
        "updated report",
        "updated report",
    ]
    assert [call["window"] for call in store.calls] == ["Inbox", "Inbox", "Thread"]


def test_daemon_resets_coalescing_after_rejected_event(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings, duplicate_window=60.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=10.0),
        _line(app="com.unknown.app", window="Other", text="not captured", ts=11.0),
        _line(app="com.apple.mail", window="Inbox", text="same report", ts=12.0),
    ]

    stats = daemon.run_lines(lines)

    assert stats.ingested == 2
    assert stats.rejected == 1
    assert stats.coalesced == 0
    assert [call["ts"] for call in store.calls] == [10.0, 12.0]


def test_daemon_rejects_non_allowlisted(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [_line(app="com.unknown.app", window="w", text="private", ts=1.0)]
    stats = daemon.run_lines(lines)
    assert stats.received == 1
    assert stats.ingested == 0
    assert stats.rejected == 1
    assert store.calls == []


def test_daemon_rejects_all_events_when_paused(allow_settings):
    pause_file = allow_settings.data_dir / "capture.paused"
    pause_file.write_text("")
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="allowed text", ts=1.0)
    ]
    stats = daemon.run_lines(lines)
    assert stats.received == 1
    assert stats.ingested == 0
    assert stats.rejected == 1
    assert store.calls == []

    pause_file.unlink()
    stats = daemon.run_lines(lines)
    assert stats.ingested == 1
    assert len(store.calls) == 1


def test_daemon_passes_pause_file_to_helper_boundary(allow_settings):
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=("capture-helper",),
        require_signed_helper=False,
    )

    argv = daemon._with_policy_args(["capture-helper"])

    assert argv[:3] == [
        "capture-helper",
        "--pause-file",
        str(allow_settings.data_dir / "capture.paused"),
    ]
    assert "--allow" in argv


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swift") is None,
    reason="capture helper requires macOS and the Swift toolchain",
)
def test_swift_helper_pause_file_exits_before_accessibility(tmp_path):
    pause_file = tmp_path / "capture.paused"
    pause_file.write_text("")

    res = subprocess.run(
        [
            "swift",
            "run",
            "--quiet",
            "CaptureHelper",
            "--no-prompt",
            "--pause-file",
            str(pause_file),
            "--allow",
            "com.apple.mail",
        ],
        cwd=ROOT / "capture-helper",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert res.stdout == ""
    assert "capture: skipped_paused" in res.stderr

    unknown = tmp_path / "missing" / "capture.paused"
    res = subprocess.run(
        [
            "swift",
            "run",
            "--quiet",
            "CaptureHelper",
            "--no-prompt",
            "--pause-file",
            str(unknown),
            "--allow",
            "com.apple.mail",
        ],
        cwd=ROOT / "capture-helper",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert res.returncode == 0, res.stderr
    assert res.stdout == ""
    assert "capture: pause_state_unknown" in res.stderr
    assert "capture: skipped_paused" in res.stderr


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swift") is None,
    reason="capture helper requires macOS and the Swift toolchain",
)
def test_swift_helper_rejects_non_pipe_stdout(tmp_path):
    """stdout pointing at a regular file is NOT a private pipe -> fail closed.

    Exercises the hardened ``stdoutIsPrivatePipe`` boundary end-to-end through the
    real helper: a redirected regular file is the canonical leak vector (captured
    content persisting to disk), so the helper must exit non-zero with a
    privacy-safe reason and write nothing to the file. No --pause-file is passed,
    so execution reaches the stdout check rather than short-circuiting on pause.
    """
    out_file = tmp_path / "stdout.txt"
    with out_file.open("wb") as fh:
        res = subprocess.run(
            [
                "swift",
                "run",
                "--quiet",
                "CaptureHelper",
                "--no-prompt",
                "--allow",
                "com.apple.mail",
            ],
            cwd=ROOT / "capture-helper",
            text=True,
            stdout=fh,  # a regular file: not a FIFO/socket
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )

    assert res.returncode == 3, res.stderr
    assert "not a private pipe" in res.stderr
    # Fail closed BEFORE any capture: nothing (content or otherwise) is written.
    assert out_file.read_bytes() == b""


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("swiftc") is None,
    reason="pure-function check requires macOS and the Swift toolchain",
)
def test_stdout_pipe_stat_decision_is_correct(tmp_path):
    """Unit-test the factored pure decision ``stdoutPipeStatIsPrivate``.

    The hardened ownership/permission logic is not reachable as distinct branches
    by driving the real helper (we cannot easily hand it a pipe owned by another
    user). So the rule is factored into a pure, I/O-free Swift function that maps
    (mode, uid, euid) -> accept/reject; we compile and assert its boundaries here.
    This mirrors the project's existing "test the factored logic" convention.

    Asserted policy (``nlink`` distinguishes a nameless kernel endpoint from a
    path-openable named one — mode bits alone cannot):
      * nameless FIFO/socket (nlink==0) + owned-by-us -> accept regardless of
        group/other bits (anon ``subprocess.PIPE`` is 0660, socketpair 0666; both
        are unreachable by path so the bits are inert);
      * named FIFO (nlink>0) owned-by-us, 0600       -> accept;
      * named FIFO (nlink>0) owned-by-us, 0660       -> REJECT (group members could
        open it by path) — the bug Codex flagged;
      * wrong type (regular file)                    -> reject;
      * owned by another uid                         -> reject.
    """
    src = ROOT / "capture-helper" / "Sources" / "CaptureHelper" / "main.swift"
    # Extract just the pure function from main.swift so the harness exercises the
    # SHIPPING source (no copy to drift), compiled standalone without AppKit/AX.
    text = src.read_text()
    start = text.index("func stdoutPipeStatIsPrivate(")
    end = text.index("\n}", start) + len("\n}")
    pure_fn = text[start:end]

    harness = f"""
import Foundation
{pure_fn}
let euid: uid_t = 501
let other: uid_t = 502
let nameless: nlink_t = 0   // anonymous pipe / socketpair
let named: nlink_t = 1      // mkfifo FIFO / bound socket
// type-only masks
let fifo = mode_t(S_IFIFO)
let sock = mode_t(S_IFSOCK)
let reg  = mode_t(S_IFREG)
func check(_ name: String, _ got: Bool, _ want: Bool) {{
    if got != want {{ FileHandle.standardError.write(Data("FAIL \\(name): got=\\(got) want=\\(want)\\n".utf8)); exit(1) }}
}}
// Nameless endpoints: owner-only check; bits are inert (no path to open by).
check("anon-pipe-0660-grouprw", stdoutPipeStatIsPrivate(mode: fifo | 0o660, uid: euid, euid: euid, nlink: nameless), true)
check("socketpair-0666", stdoutPipeStatIsPrivate(mode: sock | 0o666, uid: euid, euid: euid, nlink: nameless), true)
check("anon-pipe-other-owner", stdoutPipeStatIsPrivate(mode: fifo | 0o660, uid: other, euid: euid, nlink: nameless), false)
// Named endpoints: require owner-only bits (no group/other).
check("named-fifo-0600", stdoutPipeStatIsPrivate(mode: fifo | 0o600, uid: euid, euid: euid, nlink: named), true)
check("named-fifo-0660-grouprw-REJECT", stdoutPipeStatIsPrivate(mode: fifo | 0o660, uid: euid, euid: euid, nlink: named), false)
check("named-fifo-world-readable", stdoutPipeStatIsPrivate(mode: fifo | 0o604, uid: euid, euid: euid, nlink: named), false)
check("named-fifo-other-owner", stdoutPipeStatIsPrivate(mode: fifo | 0o600, uid: other, euid: euid, nlink: named), false)
// Wrong type is always rejected.
check("regular-file-rejected", stdoutPipeStatIsPrivate(mode: reg | 0o600, uid: euid, euid: euid, nlink: named), false)
print("ok")
"""
    harness_path = tmp_path / "pure_check.swift"
    harness_path.write_text(harness)
    bin_path = tmp_path / "pure_check"
    compile_res = subprocess.run(
        ["swiftc", str(harness_path), "-o", str(bin_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    assert compile_res.returncode == 0, compile_res.stderr
    run_res = subprocess.run(
        [str(bin_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert run_res.returncode == 0, run_res.stderr
    assert run_res.stdout.strip() == "ok"


def test_daemon_scrubs_before_ingest(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        _line(
            app="com.apple.mail",
            window="Inbox",
            text="here is my key sk-abcdefghijklmnop1234567890 ok",
            ts=5.0,
        )
    ]
    daemon.run_lines(lines)
    assert len(store.calls) == 1
    stored = store.calls[0]["text"]
    assert "sk-abcdefghijklmnop" not in stored
    assert "REDACTED" in stored


def test_daemon_skips_incognito(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        _line(
            app="com.apple.Safari",
            window="Bank (Private Browsing)",
            text="balance",
            ts=2.0,
        ),
        _line(app="com.apple.Safari", window="News", text="content", incognito=True, ts=3.0),
    ]
    stats = daemon.run_lines(lines)
    assert stats.rejected == 2
    assert store.calls == []


def test_daemon_drops_empty_after_normalization(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    # Pure chrome for Slack -> normalizes to empty -> rejected, not ingested.
    lines = [
        _line(app="com.tinyspeck.slackmacgap", window="w", text="Search\nThreads\nHuddles", ts=1.0)
    ]
    stats = daemon.run_lines(lines)
    assert stats.ingested == 0
    assert stats.rejected == 1
    assert store.calls == []


def test_daemon_malformed_line_counts_error(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        "{not valid json",
        _line(app="com.apple.mail", window="Inbox", text="good content", ts=1.0),
        "",  # blank: ignored, not an error
    ]
    stats = daemon.run_lines(lines)
    assert stats.errors == 1
    assert stats.ingested == 1


def test_daemon_isolates_store_failure(allow_settings):
    class BoomStore(FakeStore):
        def add_observation(self, *a, **k):
            raise RuntimeError("db down")

    daemon = CaptureDaemon(BoomStore(), settings=allow_settings)
    lines = [_line(app="com.apple.mail", window="Inbox", text="content", ts=1.0)]
    stats = daemon.run_lines(lines)
    assert stats.errors == 1
    assert stats.ingested == 0


# ---------------------------------------------------------------------------
# CaptureDaemon.run — real subprocess with a fake Python emitter
# ---------------------------------------------------------------------------


def test_daemon_run_with_subprocess_helper(allow_settings):
    """Drive run() against a fake helper subprocess emitting canned JSON."""
    events = [
        {"app": "com.apple.mail", "window": "Inbox", "text": "alpha report", "ts": 1.0},
        {"app": "com.unknown", "window": "w", "text": "skip me", "ts": 2.0},
        {"app": "com.apple.Safari", "window": "Docs", "text": "beta notes", "ts": 3.0},
    ]
    emitter = (
        "import json,sys\n"
        f"events={events!r}\n"
        "for e in events: sys.stdout.write(json.dumps(e)+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", emitter],
        require_signed_helper=False,
    )
    stats = daemon.run()
    assert stats.received == 3
    assert stats.ingested == 2  # mail + safari
    assert stats.rejected == 1  # unknown app
    apps = {c["app"] for c in store.calls}
    assert apps == {"com.apple.mail", "com.apple.Safari"}


def test_daemon_run_respects_max_events(allow_settings):
    emitter = (
        "import json,sys,time\n"
        "i=0\n"
        "while True:\n"
        "    sys.stdout.write(json.dumps({'app':'com.apple.mail','window':'w','text':'msg %d'%i,'ts':float(i)})+'\\n')\n"
        "    sys.stdout.flush(); i+=1; time.sleep(0.01)\n"
    )
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", emitter],
        require_signed_helper=False,
    )
    stats = daemon.run(max_events=3)
    assert stats.received == 3
    assert stats.ingested == 3


# ---------------------------------------------------------------------------
# Modern secret shapes: sk-proj, Stripe live/test, env names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret, rule",
    [
        # Modern OpenAI project / service-account keys.
        ("sk-proj-abcDEF123456_ghiJKL7890mnop", "token_prefixed"),
        ("sk-svcacct-abcDEF123456ghiJKL7890", "token_prefixed"),
        # Stripe live/test secret + restricted keys.
        ("sk_live_abcdef0123456789ABCDEF", "token_prefixed"),
        ("sk_test_abcdef0123456789ABCDEF", "token_prefixed"),
        ("rk_live_abcdef0123456789ABCDEF", "token_prefixed"),
        # Env-style assignment names.
        ("OPENAI_API_KEY=sk-realbutmaskedanyway1234567890", "secret_assignment"),
        ("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI0123456789", "secret_assignment"),
        ("DATABASE_PASSWORD=hunter2longvalue", "secret_assignment"),
        ('FOO_API_KEY="quoted secret value here"', "secret_assignment"),
        ("CLIENT_TOKEN='single-quoted-secret'", "secret_assignment"),
    ],
)
def test_scrub_masks_modern_secrets(secret, rule):
    text = f"prefix {secret} suffix"
    scrubbed, matched = redact.scrub(text)
    assert rule in matched, (secret, matched, scrubbed)
    assert "REDACTED" in scrubbed
    # The high-entropy value must not survive verbatim.
    if "=" in secret or ":" in secret:
        value = re.split(r"[:=]", secret, maxsplit=1)[1].strip().strip("\"'")
        assert value not in scrubbed
    else:
        assert secret not in scrubbed


def test_scrub_quoted_value_fully_masked():
    scrubbed, matched = redact.scrub('OPENAI_API_KEY="sk-proj-supersecretvalue123456"')
    assert "secret_assignment" in matched
    assert "supersecretvalue" not in scrubbed


def test_scrub_env_name_does_not_overmatch_plain_words():
    # A sentence mentioning a non-secret word with no assignment must survive.
    text = "Please update the README and the keymap before the demo."
    scrubbed, matched = redact.scrub(text)
    assert scrubbed == text
    assert matched == ()


# ---------------------------------------------------------------------------
# Exact bundle-id matching for allow/blocklists
# ---------------------------------------------------------------------------


def test_allowlist_is_exact_not_substring(tmp_path):
    # An app whose id merely CONTAINS an allowlisted id must NOT pass.
    s = Settings(data_dir=tmp_path, allowlist=["com.apple.mail"])
    d = redact.decide(
        app="com.evil.com.apple.mail.spoof", window="w", text="hi", settings=s
    )
    assert not d.capture
    assert d.reason == "not_allowlisted"
    # Exact id still passes.
    d2 = redact.decide(app="com.apple.mail", window="w", text="hi", settings=s)
    assert d2.capture


def test_blocklist_is_exact_not_substring(tmp_path):
    # Blocklist entry must not subtract an app that only contains it as substring.
    s = Settings(
        data_dir=tmp_path,
        allowlist=["com.apple.mailbox"],
        blocklist=["com.apple.mail"],
    )
    d = redact.decide(app="com.apple.mailbox", window="w", text="hi", settings=s)
    assert d.capture  # not blocked: exact-match blocklist doesn't catch substring


def test_allowlist_glob_entry(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=["glob:com.acme.*"])
    assert redact.decide(
        app="com.acme.editor", window="w", text="hi", settings=s
    ).capture
    # Glob matches the whole id; a different vendor with the prefix elsewhere
    # is rejected (see test_allowlist_glob_does_not_match_other_vendor).
    assert redact.decide(
        app="com.acme.editor.helper", window="w", text="hi", settings=s
    ).capture  # com.acme.* matches deeper ids by design


def test_allowlist_glob_does_not_match_other_vendor(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=["glob:com.acme.*"])
    assert not redact.decide(
        app="com.other.acme", window="w", text="hi", settings=s
    ).capture


def test_allowlist_regex_entry(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=[r"re:com\.acme\.(mail|notes)"])
    assert redact.decide(
        app="com.acme.mail", window="w", text="hi", settings=s
    ).capture
    assert redact.decide(
        app="com.acme.notes", window="w", text="hi", settings=s
    ).capture
    assert not redact.decide(
        app="com.acme.terminal", window="w", text="hi", settings=s
    ).capture


def test_malformed_regex_entry_fails_closed(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=["re:com.acme.([unclosed"])
    d = redact.decide(app="com.acme.mail", window="w", text="hi", settings=s)
    assert not d.capture  # bad regex never silently matches


# ---------------------------------------------------------------------------
# Metadata scrubbing: URL query/fragment + window titles
# ---------------------------------------------------------------------------


def test_scrub_url_strips_query_and_fragment_by_default():
    url = "https://example.com/doc?access_token=abc123&code=xyz#access_token=frag"
    out = redact.scrub_url(url)
    assert out == "https://example.com/doc"


def test_scrub_url_keep_query_redacts_sensitive_keys():
    url = "https://example.com/p?token=secret&page=3&email=a@b.com#frag"
    out = redact.scrub_url(url, keep_query=True)
    assert "secret" not in out
    assert "a@b.com" not in out
    assert "page=3" in out
    assert "REDACTED" in out
    assert "#" not in out  # fragment always dropped


def test_scrub_url_handles_non_url():
    assert redact.scrub_url("just a title") == "just a title"
    assert redact.scrub_url(None) is None
    assert redact.scrub_url("") == ""


def test_scrub_title_runs_secret_patterns():
    title = "Slack — token sk-abcdefghijklmnop1234567890"
    scrubbed, matched = redact.scrub_title(title)
    assert "token_prefixed" in matched
    assert "sk-abcdefghijklmnop" not in scrubbed


def test_scrub_metadata_combines():
    win, url, rules = redact.scrub_metadata(
        window="api_key = supersecretvalue123",
        url="https://x.com/a?code=abc#tok",
    )
    assert "supersecretvalue123" not in win
    assert url == "https://x.com/a"
    assert "secret_assignment" in rules


def test_daemon_scrubs_url_and_window_before_ingest(allow_settings):
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    lines = [
        _line(
            app="com.apple.Safari",
            window="Doc — password: hunter2secret",
            url="https://docs.example.com/d/123?access_token=leak#tok=leak2",
            text="body content",
            ts=1.0,
        )
    ]
    daemon.run_lines(lines)
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["url"] == "https://docs.example.com/d/123"
    assert "hunter2secret" not in (call["window"] or "")
    assert "leak" not in (call["url"] or "")


# ---------------------------------------------------------------------------
# Store/embed failure must not leak content into logs
# ---------------------------------------------------------------------------


def test_store_failure_does_not_log_content(allow_settings, caplog):
    captured_text = "TOP_SECRET_PAYLOAD_DO_NOT_LOG_42"

    class LeakyStore(FakeStore):
        def add_observation(self, text, **k):
            # Simulate a store/embed layer that embeds the input in its message.
            raise RuntimeError(f"db error while indexing: {text}")

    daemon = CaptureDaemon(LeakyStore(), settings=allow_settings)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text=captured_text, ts=1.0)
    ]
    with caplog.at_level("DEBUG", logger="openbird.capture"):
        stats = daemon.run_lines(lines)
    assert stats.errors == 1
    assert daemon.error_count == 1
    # The captured text must appear NOWHERE in any emitted log record, including
    # exception messages / tracebacks.
    blob = "\n".join(r.getMessage() for r in caplog.records)
    blob += "\n".join(
        (r.exc_text or "") for r in caplog.records if r.exc_text is not None
    )
    assert captured_text not in blob
    # A safe diagnostic (error type) is still present.
    assert any("RuntimeError" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Signed-helper resolution: fail closed on missing/dev binary
# ---------------------------------------------------------------------------


def test_default_helper_is_signed_bundle_path_not_dev_build(monkeypatch):
    monkeypatch.delenv("OPENBIRD_CAPTURE_HELPER", raising=False)
    cmd = default_helper_cmd()
    assert ".build/release" not in cmd[0]
    assert cmd[0].startswith("/Applications/")


def test_default_helper_honors_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "helper"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("OPENBIRD_CAPTURE_HELPER", str(fake))
    assert default_helper_cmd() == (str(fake),)


def test_run_fails_closed_when_helper_missing(allow_settings, monkeypatch):
    monkeypatch.delenv("OPENBIRD_CAPTURE_HELPER", raising=False)
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=["/nonexistent/openbird/capture-helper"],
    )
    with pytest.raises(HelperUnavailableError):
        daemon.run()


def test_run_fails_closed_when_helper_not_executable(allow_settings, tmp_path):
    not_exec = tmp_path / "helper"
    not_exec.write_text("data")
    not_exec.chmod(0o644)
    daemon = CaptureDaemon(
        FakeStore(), settings=allow_settings, helper_cmd=[str(not_exec)]
    )
    with pytest.raises(HelperUnavailableError):
        daemon.run()


def test_signed_helper_check_skipped_for_test_emitter(allow_settings):
    # require_signed_helper=False lets tests use a python emitter without a bundle.
    emitter = (
        "import json,sys\n"
        "sys.stdout.write(json.dumps("
        "{'app':'com.apple.mail','window':'w','text':'ok','ts':1.0})+'\\n')\n"
    )
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", emitter],
        require_signed_helper=False,
    )
    stats = daemon.run()
    assert stats.ingested == 1


def test_capture_cli_registers_top_level_command(monkeypatch, allow_settings):
    """Smoke-test root CLI wiring without running a real helper process."""
    from typer.testing import CliRunner

    import openbird.capture.cli as capture_cli
    import openbird.cli as root_cli
    from openbird.capture import daemon as daemon_mod
    from openbird.memory import store as store_mod

    captured: dict[str, object] = {}

    class FakeCliStore:
        def __init__(self, *, settings, provider=None):
            captured["store_settings"] = settings
            captured["store_provider"] = provider

        def close(self) -> None:
            captured["store_closed"] = True

    class FakeDaemon:
        def __init__(self, store, *, settings, helper_cmd, require_signed_helper):
            captured["store"] = store
            captured["settings"] = settings
            captured["helper_cmd"] = helper_cmd
            captured["require_signed_helper"] = require_signed_helper

        def run(self, *, max_events):
            captured["max_events"] = max_events
            return CaptureStats(received=3, ingested=2, rejected=1, errors=0)

    monkeypatch.setattr(capture_cli, "get_settings", lambda: allow_settings)
    monkeypatch.setattr(store_mod, "MemoryStore", FakeCliStore)
    monkeypatch.setattr(daemon_mod, "CaptureDaemon", FakeDaemon)
    # Capture now routes through the cloud-checked provider; stub it.
    monkeypatch.setattr(root_cli, "_provider", lambda: object())

    result = CliRunner().invoke(
        root_cli.app,
        [
            "capture",
            "--helper",
            "fake-helper --emit",
            "--allow-unsigned",
            "--max-events",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "received=3" in result.output
    assert "ingested=2" in result.output
    assert captured["helper_cmd"] == ("fake-helper", "--emit")
    assert captured["require_signed_helper"] is False
    assert captured["settings"] is allow_settings
    assert captured["store_settings"] is allow_settings
    assert captured["max_events"] == 3
    assert captured["store_closed"] is True


# ---------------------------------------------------------------------------
# stderr is drained so a chatty helper cannot deadlock capture
# ---------------------------------------------------------------------------


def test_run_does_not_deadlock_on_large_stderr(allow_settings):
    # Helper writes a large volume to stderr (>64KB pipe) while emitting events
    # on stdout. Without stderr draining this would deadlock.
    emitter = (
        "import json,sys\n"
        "sys.stderr.write('x'*500000)\n"
        "sys.stderr.flush()\n"
        "for i in range(3):\n"
        "    sys.stdout.write(json.dumps("
        "{'app':'com.apple.mail','window':'w','text':'msg %d'%i,'ts':float(i)})+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", emitter],
        require_signed_helper=False,
    )
    stats = daemon.run()
    assert stats.received == 3
    assert stats.ingested == 3


# ---------------------------------------------------------------------------
# supervised continuous-capture loop (run_forever)
# ---------------------------------------------------------------------------


def _oneshot_emitter(n: int) -> list[str]:
    """A helper cmd: prints n events then EXITS (like the real one-shot helper)."""
    code = (
        "import json,sys\n"
        f"for i in range({n}):\n"
        "    sys.stdout.write(json.dumps({'app':'com.apple.mail','window':'w',"
        "'text':'msg %d'%i,'ts':float(i)})+'\\n')\n"
        "sys.stdout.flush()\n"
    )
    return [sys.executable, "-c", code]


def test_run_forever_respawns_helper_each_cycle(allow_settings):
    # The one-shot helper emits 2 events then exits; run_forever must
    # re-spawn it each cycle, so 3 cycles ingest 3x the events.
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=_oneshot_emitter(2),
        require_signed_helper=False,
        duplicate_window=0,
    )
    stats = daemon.run_forever(poll_interval=0.0, max_cycles=3)
    assert stats.received == 6  # 3 re-spawns x 2 events — proves continuity
    assert stats.ingested == 6


def test_run_forever_stops_when_event_already_set(allow_settings):
    # A pre-set stop event means the loop body never runs (clean no-op).
    import threading

    stop = threading.Event()
    stop.set()
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=_oneshot_emitter(2),
        require_signed_helper=False,
    )
    stats = daemon.run_forever(poll_interval=0.0, stop_event=stop, max_cycles=99)
    assert stats.received == 0


def test_run_forever_circuit_breaker_trips_on_repeated_failure(
    allow_settings, monkeypatch
):
    # Consecutive failing cycles must trip the breaker and stop, not spin.
    import openbird.capture.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "_BACKOFF_BASE", 0.0)  # no real sleeps
    store = FakeStore()
    daemon = CaptureDaemon(
        store,
        settings=allow_settings,
        helper_cmd=_oneshot_emitter(1),
        require_signed_helper=False,
    )

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("transient helper failure")

    monkeypatch.setattr(daemon, "run", _boom)
    with pytest.raises(CaptureSupervisorError):
        daemon.run_forever(poll_interval=0.0, max_consecutive_failures=3)
    assert calls["n"] == 3  # stopped exactly at the breaker threshold
    assert daemon.error_count == 3


def test_run_forever_propagates_helper_unavailable(allow_settings):
    # A missing signed bundle is permanent, not transient: re-raise, don't
    # retry forever.
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=["/nonexistent/openbird/capture-helper"],
        require_signed_helper=True,
    )
    with pytest.raises(HelperUnavailableError):
        daemon.run_forever(poll_interval=0.0, max_cycles=5)


def test_run_forever_resets_failure_count_after_success(allow_settings, monkeypatch):
    # A success between failures must reset the consecutive-failure counter
    # so the breaker only trips on *consecutive* failures.
    import openbird.capture.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "_BACKOFF_BASE", 0.0)
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=_oneshot_emitter(1),
        require_signed_helper=False,
    )
    seq = iter([RuntimeError("f1"), RuntimeError("f2"), None, RuntimeError("f3")])
    real_run = daemon.run

    def _maybe_boom(*a, **k):
        try:
            exc = next(seq)
        except StopIteration:
            return real_run(*a, **k)
        if exc is not None:
            raise exc
        return real_run(*a, **k)

    monkeypatch.setattr(daemon, "run", _maybe_boom)
    # 2 fails, 1 success (resets), 1 fail, then clean cycles. Breaker=3 never trips.
    stats = daemon.run_forever(poll_interval=0.0, max_consecutive_failures=3, max_cycles=3)
    assert stats.received >= 1  # at least the successful + subsequent cycles ran


def test_run_forever_nonzero_helper_exit_trips_breaker(allow_settings, monkeypatch):
    # A helper exiting non-zero ON ITS OWN (e.g. Accessibility denied=2)
    # must count as a failure so the breaker trips — not be re-spawned forever.
    import openbird.capture.daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "_BACKOFF_BASE", 0.0)
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", "import sys; sys.exit(2)"],
        require_signed_helper=False,
    )
    with pytest.raises(CaptureSupervisorError):
        daemon.run_forever(poll_interval=0.0, max_consecutive_failures=3)
    assert daemon.error_count == 3  # each non-zero exit counted, breaker tripped


def test_run_clean_exit_zero_is_not_a_failure(allow_settings):
    # A helper that exits 0 after emitting is a success, not a failure.
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=_oneshot_emitter(1),
        require_signed_helper=False,
    )
    stats = daemon.run()  # must not raise HelperExitError
    assert stats.received == 1


def test_run_terminates_helper_when_stop_set(allow_settings):
    # With stop set, an active/hung helper is terminated promptly so a
    # clean shutdown isn't blocked on the stdout iterator (no hang, no raise).
    import threading

    code = (
        "import json,sys,time\n"
        "sys.stdout.write(json.dumps({'app':'com.apple.mail','window':'w',"
        "'text':'x','ts':1.0})+'\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(3600)\n"  # hang forever unless terminated
    )
    stop = threading.Event()
    stop.set()
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", code],
        require_signed_helper=False,
    )
    stats = daemon.run(stop_event=stop)  # returns promptly; does not hang/raise
    assert isinstance(stats, CaptureStats)


def test_run_stop_initiated_exit_not_misclassified_as_failure(allow_settings):
    # A helper that traps SIGTERM and exits positive after WE stop
    # it must NOT raise HelperExitError — the stop was our-initiated.
    import threading

    code = (
        "import signal,sys,time\n"
        "signal.signal(signal.SIGTERM, lambda *a: sys.exit(7))\n"
        "sys.stdout.write('\\n'); sys.stdout.flush()\n"  # one blank line, then hang
        "time.sleep(3600)\n"
    )
    stop = threading.Event()
    stop.set()
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", code],
        require_signed_helper=False,
    )
    # Must return cleanly (no HelperExitError) even though the child exits 7.
    stats = daemon.run(stop_event=stop)
    assert isinstance(stats, CaptureStats)


def test_run_signal_killed_helper_is_a_failure(allow_settings):
    # A helper that dies by signal (negative returncode, e.g. SIGKILL=-9)
    # during normal operation is a genuine failure, not a clean exit.
    code = "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"
    daemon = CaptureDaemon(
        FakeStore(),
        settings=allow_settings,
        helper_cmd=[sys.executable, "-c", code],
        require_signed_helper=False,
    )
    with pytest.raises(HelperExitError):
        daemon.run()


# ---------------------------------------------------------------------------
# dangerous-app list parity: the canonical JSON, the Swift baked fallback,
# and the Python baked tuple MUST stay in lockstep (single source of truth).
# ---------------------------------------------------------------------------

# tests/unit/test_capture.py -> repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DANGEROUS_JSON = (
    _REPO_ROOT / "capture-helper" / "Sources" / "CaptureHelper" / "dangerous_apps.json"
)
_CAPTURE_SWIFT = (
    _REPO_ROOT / "capture-helper" / "Sources" / "CaptureHelper" / "main.swift"
)


def _json_dangerous_set() -> set[str]:
    data = json.loads(_DANGEROUS_JSON.read_text())
    return {s.lower() for s in data["dangerous_bundle_substrings"]}


def _swift_fallback_set() -> set[str]:
    """Parse the Swift `dangerousBundleSubstrings` baked array literal."""
    src = _CAPTURE_SWIFT.read_text()
    m = re.search(
        r"dangerousBundleSubstrings\s*:\s*\[String\]\s*=\s*\[(.*?)\]",
        src,
        re.DOTALL,
    )
    assert m, "could not locate dangerousBundleSubstrings literal in main.swift"
    return {tok.lower() for tok in re.findall(r'"([^"]+)"', m.group(1))}


def test_dangerous_list_parity_json_swift_python():
    json_set = _json_dangerous_set()
    swift_set = _swift_fallback_set()
    python_set = {s.lower() for s in redact._DANGEROUS_BUNDLE_SUBSTRINGS}

    # Pairwise diffs for diagnosis (so a failure says exactly what drifted).
    assert json_set == python_set, (
        f"JSON vs Python drift: only-json={json_set - python_set} "
        f"only-python={python_set - json_set}"
    )
    assert json_set == swift_set, (
        f"JSON vs Swift drift: only-json={json_set - swift_set} "
        f"only-swift={swift_set - json_set}"
    )
    assert swift_set == python_set, (
        f"Swift vs Python drift: only-swift={swift_set - python_set} "
        f"only-python={python_set - swift_set}"
    )

    # The backstop must never be empty and must contain the historically-listed
    # vendors from BOTH original sides (the union the drift was hiding).
    for vendor in ("1password", "keychain", "keeper", "protonpass", "keychainaccess"):
        assert vendor in python_set


# ---------------------------------------------------------------------------
# Layer 1 — volatility.normalize: de-flicker animated UI noise (conservative)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("⠂ Building project", "Building project"),          # braille spinner
        ("⠿⠿ Loading", "Loading"),                            # repeated braille
        ("✳ Thinking…", "Thinking…"),                         # star "thinking" glyph
        ("\x1b[32mgreen\x1b[0m text", "green text"),          # ANSI SGR stripped
        ("Downloading model-x [####   ] 45%", "Downloading model-x"),  # wide bracket bar
        ("install [==========]", "install"),                  # wide bar, no percent
        ("step [## ] 5%", "step"),                             # short body but has percent
        ("loss 37%|███      |", "loss"),                       # tqdm pipe bar
        ("progress ███████░░░ done", "progress done"),        # bare block-bar run
        ("|", ""),                                             # whole-line ASCII spinner
        ("\\", ""),
    ],
)
def test_volatility_strips_volatile_tokens(raw, expected):
    assert volatility.normalize(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "- bullet item",            # markdown bullet (leading '-' must survive)
        "* list item",
        "/usr/bin/env python",      # path (leading '/' must survive)
        "| pipe | table |",         # pipes that are not a progress bar
        "[TODO] finish this",       # bracket without bar chars
        "[1.2.3] version tag",
        "[ERROR] something failed",
        "rebase marker [#] here",   # short bracket markers are NOT bars...
        "arrow [=>] points",        # ...require width-or-percent to strip
        "ascii [==>] flow",
        "single [>] gt",
        "tag [=] eq",
        "[----] dashes only",       # dashes are bar-body but not a bar SIGNAL
        "10:15:30 build failed",    # bare clock — DEFERRED, must survive
        "API returned in (12s)",    # parenthesized elapsed — DEFERRED, must survive
        "see https://example.com/path",
        "plain prose with no noise",
    ],
)
def test_volatility_preserves_meaningful_text(text):
    # Conservative: never rewrite real content (paths, bullets, timestamps, prose).
    assert volatility.normalize(text) == text


def test_volatility_is_idempotent():
    sample = "⠂ Building [##   ] 20%\n- keep this line\n\x1b[31mred\x1b[0m\n37%|██  |"
    once = volatility.normalize(sample)
    assert volatility.normalize(once) == once


def test_volatility_progress_bar_keeps_distinct_labels():
    # Two different downloads must NOT collapse (label differs); the same download
    # progressing MUST collapse (only the bar/percent differs).
    a1 = volatility.normalize("Downloading model-x [#    ] 5%")
    a2 = volatility.normalize("Downloading model-x [####] 80%")
    b1 = volatility.normalize("Downloading model-y [#    ] 5%")
    assert a1 == a2  # same download, different frame -> identical
    assert a1 != b1  # different downloads stay distinct


def test_volatility_pure_spinner_line_becomes_empty():
    assert volatility.normalize("⠹").strip() == ""


# ---------------------------------------------------------------------------
# Layer 1 — spinner-frame bloat regression (the load-bearing test)
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = "⠂⠠⠐⠈⠁⠉⠙⠹"


def _ghostty_settings(tmp_path) -> Settings:
    """Allow Ghostty + clear the blocklist so a terminal can be driven in-test."""
    return Settings(
        data_dir=tmp_path,
        allowlist=["com.mitchellh.ghostty"],
        blocklist=[],
    )


def test_spinner_frames_coalesce_to_single_observation(tmp_path):
    # N frames identical except a rotating spinner glyph must de-flicker to one
    # capture: the coalesce gate now matches across frames (was: N new blobs).
    store = FakeStore()
    settings = _ghostty_settings(tmp_path)
    daemon = CaptureDaemon(store, settings=settings, duplicate_window=60.0)
    body = "Compiling project...\n  module foo\n  module bar\nrunning 42 tests"
    lines = [
        _line(app="com.mitchellh.ghostty", window="term", text=f"{g} {body}", ts=float(i))
        for i, g in enumerate(_SPINNER_FRAMES)
    ]

    stats = daemon.run_lines(lines)

    assert stats.received == len(_SPINNER_FRAMES)
    assert stats.ingested == 1
    assert stats.coalesced == len(_SPINNER_FRAMES) - 1
    assert len(store.calls) == 1
    # The stored text is de-flickered (no spinner glyph survived).
    assert "⠂" not in store.calls[0]["text"]
    assert "Compiling project" in store.calls[0]["text"]


def test_spinner_frames_dedupe_to_single_blob_in_real_store(tmp_path):
    # Against a REAL store with the daemon coalesce gate DISABLED (duplicate_window
    # =0), every de-flickered frame reaches add_observation, proving blob-level
    # dedup (content_blobs) — the actual 300 MB cause. WITHOUT Layer 1 the rotating
    # glyph would fork content_hash and yield one 184 KB blob PER frame.
    settings = Settings(
        data_dir=tmp_path,
        embed_dim=64,
        allowlist=["com.mitchellh.ghostty"],
        blocklist=[],
    )
    store = MemoryStore(
        db_path=str(tmp_path / "bloat.db"),
        settings=settings,
        provider=FakeProvider(embed_dim=64),
    )
    try:
        daemon = CaptureDaemon(store, settings=settings, duplicate_window=0)
        body = "Compiling project...\n  module foo\n  module bar\nrunning 42 tests"
        lines = [
            _line(
                app="com.mitchellh.ghostty",
                window="term",
                text=f"{g} {body}",
                ts=float(i),
            )
            for i, g in enumerate(_SPINNER_FRAMES)
        ]

        stats = daemon.run_lines(lines)

        assert stats.ingested == len(_SPINNER_FRAMES)  # coalesce gate disabled
        st = store.stats()
        assert st["observations"] == len(_SPINNER_FRAMES)  # timeline preserved
        assert st["blobs"] == 1  # de-flickered text collapses to ONE blob
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Layer 3 — terminals blocked by default (recovery is blocklist-override)
# ---------------------------------------------------------------------------


def test_terminal_blocked_even_when_also_allowlisted(tmp_path):
    # Blocklist is subtractive: a default-blocked terminal stays blocked even if a
    # user ALSO allowlists it. Allowlisting alone is NOT the recovery path.
    s = Settings(data_dir=tmp_path, allowlist=["com.mitchellh.ghostty"])
    d = redact.decide(
        app="com.mitchellh.ghostty", window="term", text="secrets here", settings=s
    )
    assert not d.capture
    assert d.reason == "blocklisted"


def test_terminal_recovery_requires_blocklist_override(tmp_path):
    # Real recovery: drop the terminal from the blocklist AND allowlist it.
    s = Settings(
        data_dir=tmp_path,
        allowlist=["com.mitchellh.ghostty"],
        blocklist=[],
    )
    d = redact.decide(
        app="com.mitchellh.ghostty", window="term", text="content", settings=s
    )
    assert d.capture
    assert d.reason == "allowlisted"


def test_default_blocklist_covers_third_party_terminals(tmp_path):
    # A non-terminal allowlisted app still captures (no over-blocking).
    s = Settings(data_dir=tmp_path, allowlist=["com.apple.mail"])
    for term in ("net.kovidgoyal.kitty", "org.alacritty", "com.github.wez.wezterm"):
        d = redact.decide(app=term, window="w", text="x", settings=s)
        assert not d.capture, term
    assert redact.decide(
        app="com.apple.mail", window="Inbox", text="hi", settings=s
    ).capture


# ---------------------------------------------------------------------------
# Layer 4 — episodic session segmentation
# ---------------------------------------------------------------------------


def test_session_continues_within_gap_and_same_app(allow_settings):
    store = FakeStore()
    settings = Settings(
        data_dir=allow_settings.data_dir,
        allowlist=list(allow_settings.allowlist),
        session_gap_seconds=300.0,
    )
    daemon = CaptureDaemon(store, settings=settings, duplicate_window=0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="first note", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="second note", ts=120.0),
    ]
    daemon.run_lines(lines)
    sessions = [c["session_id"] for c in store.calls]
    assert len(sessions) == 2
    assert sessions[0] is not None
    assert sessions[0] == sessions[1]  # same app, within gap -> one session


def test_session_breaks_on_app_switch_and_on_gap(allow_settings):
    store = FakeStore()
    settings = Settings(
        data_dir=allow_settings.data_dir,
        allowlist=list(allow_settings.allowlist),
        session_gap_seconds=100.0,
    )
    daemon = CaptureDaemon(store, settings=settings, duplicate_window=0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="a", ts=10.0),
        _line(app="com.apple.Safari", window="Docs", text="b", ts=20.0),   # app switch
        _line(app="com.apple.Safari", window="Docs", text="c", ts=400.0),  # > gap jump
    ]
    daemon.run_lines(lines)
    s = [c["session_id"] for c in store.calls]
    assert len(s) == 3
    assert s[0] != s[1]  # app switch -> new session
    assert s[1] != s[2]  # gap jump -> new session


def test_session_continuity_survives_coalesced_heartbeat(allow_settings):
    # The session-activity clock must advance on a COALESCED frame, so continuity
    # is independent of duplicate_window. With duplicate_window >> session_gap, a
    # coalesced heartbeat at t=80 keeps the gap-clock fresh; the t=150 ingest then
    # stays in the SAME session. If the clock only advanced on ingest (buggy), the
    # gap from t=10 to t=150 (140 > 100) would spuriously start a new session.
    store = FakeStore()
    settings = Settings(
        data_dir=allow_settings.data_dir,
        allowlist=list(allow_settings.allowlist),
        session_gap_seconds=100.0,
    )
    daemon = CaptureDaemon(store, settings=settings, duplicate_window=1000.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same", ts=10.0),   # ingest
        _line(app="com.apple.mail", window="Inbox", text="same", ts=80.0),   # coalesced
        _line(app="com.apple.mail", window="Inbox", text="changed", ts=150.0),  # ingest
    ]
    stats = daemon.run_lines(lines)
    assert stats.ingested == 2
    assert stats.coalesced == 1
    sessions = [c["session_id"] for c in store.calls]
    assert len(sessions) == 2
    assert sessions[0] == sessions[1]  # continuity held across the coalesced frame


def test_session_clock_does_not_regress_on_backward_frame(allow_settings):
    # A backward / out-of-order frame must NOT rewind the session-activity clock.
    # Sequence (gap=100, big duplicate_window so repeats coalesce):
    #   t=10  ingest  -> session S1, clock=10
    #   t=80  coalesce-> clock advances to 80
    #   t=50  coalesce-> BACKWARD; clock must stay 80 (not regress to 50)
    #   t=160 ingest  -> gap from 80 is 80 (<100) => still S1
    # If the clock regressed to 50, t=160 would see gap 110 (>100) -> new session.
    store = FakeStore()
    settings = Settings(
        data_dir=allow_settings.data_dir,
        allowlist=list(allow_settings.allowlist),
        session_gap_seconds=100.0,
    )
    daemon = CaptureDaemon(store, settings=settings, duplicate_window=1000.0)
    lines = [
        _line(app="com.apple.mail", window="Inbox", text="same", ts=10.0),
        _line(app="com.apple.mail", window="Inbox", text="same", ts=80.0),
        _line(app="com.apple.mail", window="Inbox", text="same", ts=50.0),
        _line(app="com.apple.mail", window="Inbox", text="changed", ts=160.0),
    ]
    daemon.run_lines(lines)
    sessions = [c["session_id"] for c in store.calls]
    assert len(sessions) == 2  # t=10 and t=160 ingested (others coalesced)
    assert sessions[0] == sessions[1]  # one continuous session; no spurious split


# ---------------------------------------------------------------------------
# Defense-in-depth: re-scrub after de-flicker; non-finite timestamp sanitizing
# ---------------------------------------------------------------------------


def test_ansi_split_secret_is_scrubbed_after_deflicker(allow_settings):
    # An ANSI escape splitting a token slips past the FIRST scrub (the escape
    # breaks the regex), but volatility strips the escape and the RE-scrub masks
    # the rejoined key — so a de-flickered secret can never be stored.
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    # Named ``key`` (not ``secret``) to match the existing test fixtures in this
    # file and avoid Ruff S105 (hardcoded-password) on an intentional fixture.
    key = "sk-abcdefghijklmnop1234567890"
    split = "sk-abcdefgh\x1b[31mijklmnop1234567890"  # ESC SGR mid-token
    lines = [_line(app="com.apple.mail", window="Inbox", text=f"key {split} end", ts=1.0)]
    daemon.run_lines(lines)
    assert len(store.calls) == 1
    stored = store.calls[0]["text"]
    assert key not in stored
    assert "ijklmnop1234567890" not in stored
    assert "REDACTED" in stored


def test_event_clock_sanitizes_non_finite_ts():
    assert CaptureDaemon._event_clock(10.0) == 10.0
    assert CaptureDaemon._event_clock(None) > 0  # fallback to wall clock
    for bad in (float("nan"), float("inf"), float("-inf"), "not-a-number"):
        assert math.isfinite(CaptureDaemon._event_clock(bad))


def test_nan_timestamp_frame_stores_finite_ts(allow_settings):
    # A NaN ts (valid JSON literal) must not reach storage as NaN — it would
    # corrupt time-range ordering and freeze session segmentation.
    store = FakeStore()
    daemon = CaptureDaemon(store, settings=allow_settings)
    line = _line(app="com.apple.mail", window="Inbox", text="hello there", ts=float("nan"))
    stats = daemon.run_lines([line])
    assert stats.ingested == 1
    assert math.isfinite(store.calls[0]["ts"])
