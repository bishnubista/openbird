"""Unit tests for redact.py's self-capture backstop.

OpenBird must NEVER ingest its own UI: capturing the "Ask about your work…"
window would poison memory with phantom rows and create a retrieval feedback
loop. The backstop (:func:`redact._is_self_capture` + the early gate in
:func:`redact.decide`) matches OpenBird's own bundle id exactly or as a dotted
child (capture-helper / audio-helper), case-insensitively, and **never** by
substring.

The substring exclusion is the critical regression guard: the literal string
"openbird" appears in ~1,459 legitimate Chrome rows
(github.com/bishnubista/openbird) and ~1,290 Ghostty build rows on the real DB,
so a substring match would silently delete that genuine dev signal.

These tests are additive — they exercise the self-capture path that the existing
``tests/unit/test_capture.py`` allowlist/blocklist suite does not cover.
"""

from __future__ import annotations

import pytest

from openbird.capture import redact
from openbird.config import Settings


# ---------------------------------------------------------------------------
# _is_self_capture — exact / dotted-prefix, case-insensitive, NEVER substring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "app",
    [
        "ai.openbird.openbird",  # exact root (lowercase)
        "ai.openbird.OpenBird",  # exact root (canonical casing)
        "AI.OPENBIRD.OPENBIRD",  # exact root (uppercase) — case-insensitive
        "ai.openbird.OpenBird.capture-helper",  # bundled helper (dotted child)
        "ai.openbird.OpenBird.audio-helper",  # bundled helper (dotted child)
        "ai.openbird.openbird.AUDIO-HELPER",  # helper, mixed case
    ],
)
def test_is_self_capture_true_for_own_app_and_helpers(app):
    assert redact._is_self_capture(app) is True


@pytest.mark.parametrize(
    "app",
    [
        "com.google.Chrome",  # a wholly unrelated app
        "com.mitchellh.ghostty",  # terminal building openbird
        None,  # missing app id never matches
        "",  # empty app id never matches
        "com.acme.openbirdish",  # SUBSTRING trap — must NOT match
        "com.acme.openbird",  # "openbird" as a leaf elsewhere — not ours
        "ai.openbird.openbirder",  # shares the root as a PREFIX but no dot
        "openbird",  # bare token — not the dotted bundle root
        "ai.openbird.openbirdX",  # root immediately followed by non-dot
    ],
)
def test_is_self_capture_false_for_non_self_apps(app):
    assert redact._is_self_capture(app) is False


def test_self_bundle_root_constant_is_lowercased():
    # The constant the matcher compares against (via casefold) is the canonical
    # lowercase root; documents the SINGLE source of truth for the match.
    assert redact._SELF_BUNDLE_ROOT == "ai.openbird.openbird"


def test_self_bundle_root_swift_python_parity():
    """The Swift helper's `selfBundleRoot` MUST equal `redact._SELF_BUNDLE_ROOT`.

    Phase C2 added a helper-side self-capture gate (before the allowlist/AX/
    SCK, so OpenBird's own UI is never read or screenshotted at the source).
    Two-copy parity, same spirit as the dangerous-list tri-source test — there
    is no JSON resource here, just the one constant on each side.
    """
    import re
    from pathlib import Path

    main_swift = (
        Path(__file__).resolve().parents[2]
        / "capture-helper" / "Sources" / "CaptureHelper" / "main.swift"
    ).read_text()
    m = re.search(r'let\s+selfBundleRoot\s*=\s*"([^"]+)"', main_swift)
    assert m, "could not locate selfBundleRoot literal in main.swift"
    assert m.group(1) == redact._SELF_BUNDLE_ROOT


def test_is_self_capture_rejects_substring_regression_guard():
    # The load-bearing guard: a substring match would delete ~1,459 legitimate
    # Chrome rows and ~1,290 Ghostty rows whose text/ids contain "openbird".
    assert redact._is_self_capture("com.acme.openbirdish") is False
    # ...yet the real self ids still match.
    assert redact._is_self_capture("ai.openbird.OpenBird") is True


# ---------------------------------------------------------------------------
# decide() — self-capture gate fires BEFORE the allowlist
# ---------------------------------------------------------------------------


def test_decide_self_capture_under_default_empty_allowlist(tmp_path):
    # Default first-run posture: empty allowlist captures nothing. An OpenBird
    # self event must still report the precise ``self_capture`` reason (the gate
    # runs before the allowlist check), not the generic ``not_allowlisted``.
    s = Settings(data_dir=tmp_path, allowlist=[])
    d = redact.decide(
        app="ai.openbird.OpenBird",
        window="Ask about your work…",
        text="what did I work on today?",
        settings=s,
    )
    assert not d.capture
    assert d.reason == "self_capture"


def test_decide_self_capture_helper_under_empty_allowlist(tmp_path):
    s = Settings(data_dir=tmp_path, allowlist=[])
    d = redact.decide(
        app="ai.openbird.OpenBird.capture-helper",
        window="helper",
        text="some captured text",
        settings=s,
    )
    assert not d.capture
    assert d.reason == "self_capture"


def test_decide_self_capture_even_when_explicitly_allowlisted(tmp_path):
    # Even if OpenBird were (mis)allowlisted, the self gate runs first and wins,
    # so it can never ingest its own UI.
    s = Settings(data_dir=tmp_path, allowlist=["ai.openbird.OpenBird"])
    d = redact.decide(
        app="ai.openbird.OpenBird",
        window="OpenBird",
        text="phantom ask row",
        settings=s,
    )
    assert not d.capture
    assert d.reason == "self_capture"


def test_decide_self_capture_helper_even_when_root_allowlisted(tmp_path):
    # Allowlisting the root must not open a hole for the dotted helper either.
    s = Settings(data_dir=tmp_path, allowlist=["ai.openbird.OpenBird"])
    d = redact.decide(
        app="ai.openbird.OpenBird.audio-helper",
        window="audio",
        text="helper output",
        settings=s,
    )
    assert not d.capture
    assert d.reason == "self_capture"


# ---------------------------------------------------------------------------
# decide() — real dev signal (Chrome on the OpenBird repo) is NOT self-capture
# ---------------------------------------------------------------------------


def test_decide_chrome_on_openbird_repo_not_dropped_as_self(tmp_path):
    # The exact real-world false-positive vector: a Chrome window whose title
    # contains "openbird". It must follow NORMAL allowlist rules, not be dropped
    # as self_capture. With Chrome allowlisted it is captured.
    s = Settings(data_dir=tmp_path, allowlist=["com.google.Chrome"])
    d = redact.decide(
        app="com.google.Chrome",
        window="Pull requests · bishnubista/openbird",
        text="Review the diff for the self-capture backstop",
        settings=s,
    )
    assert d.capture
    assert d.reason == "allowlisted"


def test_decide_chrome_on_openbird_repo_follows_allowlist_when_not_allowed(tmp_path):
    # Same Chrome/openbird window, but Chrome is NOT allowlisted: it is rejected
    # with the ordinary ``not_allowlisted`` reason — crucially NOT ``self_capture``
    # (which would mark it for deletion as our own UI).
    s = Settings(data_dir=tmp_path, allowlist=["com.apple.mail"])
    d = redact.decide(
        app="com.google.Chrome",
        window="Pull requests · bishnubista/openbird",
        text="Review the diff for the self-capture backstop",
        settings=s,
    )
    assert not d.capture
    assert d.reason == "not_allowlisted"
