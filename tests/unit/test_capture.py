"""Unit tests for the capture subsystem: redaction, adapters, daemon.

These tests use a FAKE capture helper (canned JSON, either as in-memory lines or
a tiny ``python -c`` subprocess emitter) so no real Accessibility access, signed
bundle, or Ollama is required. The ingest sink is a lightweight fake recording
``add_observation`` calls.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from openbird.capture import adapters, redact
from openbird.capture.daemon import (
    CaptureDaemon,
    CaptureStats,
    HelperUnavailableError,
    default_helper_cmd,
    parse_event,
)
from openbird.config import Settings
from openbird.types import Observation


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
# H5 — credit-card (PAN) redaction: Luhn-validated, group-anchored.
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
    # R3 note: separated runs whose TOTAL exceeds 19 digits — the longest VALID
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
# H6 — token/JWT boundaries must not rely on Unicode-aware \b.
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
# [Finding 5] Modern secret shapes: sk-proj, Stripe live/test, env names
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
# [Finding 4] Exact bundle-id matching for allow/blocklists
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
# [Finding 6] Metadata scrubbing: URL query/fragment + window titles
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
# [Finding 1] Store/embed failure must not leak content into logs
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
# [Finding 3] Signed-helper resolution: fail closed on missing/dev binary
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
        def __init__(self, *, settings):
            captured["store_settings"] = settings

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
# [Finding 2] stderr is drained so a chatty helper cannot deadlock capture
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
# H7 — dangerous-app list parity: the canonical JSON, the Swift baked fallback,
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
