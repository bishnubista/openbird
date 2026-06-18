"""Privacy redaction for captured screen text — defense-in-depth, NOT a guarantee.

The real privacy protection in OpenBird is the **allowlist-only first run** [R3]:
nothing is captured unless its app bundle id is explicitly allowlisted, and a
blocklist (password managers, finance/health apps, terminals, editors, browsers
until enabled) is always honored. The regex secret-scrubbing here is a second
layer of defense applied to text that already passed the app gate — it cannot
catch every secret and must never be presented as a guarantee.

The pipeline for a single capture event is:

    decide(event) -> RedactionDecision   # capture? why / why not?
    scrub(text)   -> str                 # mask obvious secrets in captured text

``decide`` is allowlist-first: an app is only captured if it appears on the
allowlist. The blocklist and incognito/private signals can only ever *subtract*
from that, never add. This ordering means a misconfigured blocklist can never
accidentally start capturing a non-allowlisted app.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from openbird.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Secret-scrubbing patterns (defense-in-depth; intentionally conservative).
#
# Each entry is (name, compiled-regex, replacement). These run over text that
# has ALREADY passed the app allow/block gate. They are best-effort: regexes
# cannot reliably identify every secret, and over-aggressive masking would
# destroy legitimate content, so we target high-confidence, high-impact shapes.
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # Common API / token prefixes (OpenAI, GitHub, Slack, Stripe, AWS, Google).
    #
    # OpenAI: classic ``sk-...`` AND modern project/service keys
    # ``sk-proj-...`` / ``sk-svcacct-...`` (which embed ``-`` and ``_``).
    # Stripe: live/test secret + restricted keys ``sk_live_`` / ``sk_test_`` /
    # ``rk_live_`` / ``rk_test_``. GitHub, Slack, AWS, Google as before.
    (
        "token_prefixed",
        re.compile(
            # H6 fix: do NOT rely on Python's Unicode-aware ``\b``. A key abutting
            # non-ASCII text (e.g. ``keyσsk-...``) has no word boundary because the
            # Greek letter is itself a word char, so the real key would survive.
            # Use an explicit ASCII-class negative lookbehind/lookahead instead.
            # The LEADING boundary class deliberately EXCLUDES ``_`` so that a key
            # glued behind an underscore (``_sk-...``) still redacts; the TRAILING
            # class keeps ``_`` so a trailing ``_`` does not split the value. The
            # lookarounds wrap the WHOLE alternation (never per-branch) so ordered
            # alternation precedence is preserved.
            r"(?<![A-Za-z0-9-])(?:"
            # Stripe-style underscore keys must come before the bare ``sk-`` so
            # ``sk_live_...`` is matched as a unit (alternation is ordered).
            r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"
            # OpenAI modern project/service-account keys (contain - and _).
            r"|sk-(?:proj|svcacct|admin)-[A-Za-z0-9_\-]{16,}"
            # OpenAI classic keys.
            r"|sk-[A-Za-z0-9]{16,}"
            r"|gh[pousr]_[A-Za-z0-9]{20,}"
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"
            r"|AKIA[0-9A-Z]{12,}"
            r"|AIza[0-9A-Za-z_\-]{20,})(?![A-Za-z0-9_-])"
        ),
        "[REDACTED:token]",
    ),
    # Bearer / authorization headers.
    (
        "bearer",
        re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)\S+"),
        r"\1[REDACTED:token]",
    ),
    # JWT-shaped tokens (three base64url segments). Ordered BEFORE the generic
    # ``secret_assignment`` so that ``token=<jwt>`` is classified as a JWT (more
    # specific) rather than a bare assignment — both mask the value either way.
    (
        "jwt",
        # H6 fix: same ASCII-class boundaries as ``token_prefixed`` (see above) so
        # a JWT abutting non-ASCII text is still caught. Leading class excludes
        # ``_`` so ``_eyJ...`` redacts; the dotted base64url body keeps ``\b``-free.
        re.compile(
            r"(?<![A-Za-z0-9-])"
            r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
            r"(?![A-Za-z0-9_-])"
        ),
        "[REDACTED:jwt]",
    ),
    # PEM private-key blocks (also more specific than the generic assignment).
    (
        "pem_key",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED:private-key]",
    ),
    # key = value / password: value assignments for sensitive-looking names.
    #
    # Covers env-style upper-snake names (``OPENAI_API_KEY=``, ``FOO_API_KEY=``,
    # ``DATABASE_PASSWORD=``), conventional names (``password``, ``secret``,
    # ``client_secret``…), and tolerates an optional ``export`` prefix and
    # single/double-quoted values so the whole quoted secret is masked. Runs
    # after the structured-token rules above so they win when both could match.
    (
        "secret_assignment",
        re.compile(
            r"(?im)"
            r"(?:^|(?<=[\s;&|]))(?:export\s+)?"
            r"("
            # Any *_KEY / *_TOKEN / *_SECRET / *_PASSWORD / *_PASSWD env names,
            # e.g. OPENAI_API_KEY, AWS_SECRET_ACCESS_KEY, DB_PASSWORD.
            r"[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)*"
            r"(?:[_-](?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD))"
            # Conventional lower/mixed-case names.
            r"|password|passwd|pwd|secret|api[_-]?key|access[_-]?token"
            r"|refresh[_-]?token|client[_-]?secret|private[_-]?key|token"
            r")"
            r"(\s*[:=]\s*)"
            # Value: a single/double-quoted string, or an unquoted run.
            r"""(?:"[^"]*"|'[^']*'|\S+)"""
        ),
        r"\1\2[REDACTED]",
    ),
    # Credit-card numbers are handled separately by ``_scrub_cards`` (Luhn-
    # validated, group-anchored) because a simple regex both over-redacts long
    # numeric IDs and under-redacts 17-19 digit PANs. See ``_scrub_cards``.
    # US Social Security numbers.
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED:ssn]",
    ),
]

# Window-title / app-name substrings that strongly indicate a private or
# incognito browsing context. Matched case-insensitively against the window
# title. These can only *block* capture, never enable it.
_INCOGNITO_MARKERS: tuple[str, ...] = (
    "incognito",
    "private browsing",
    "inprivate",
    "private window",
)

# Bundle-id prefixes for app categories that are blocked by default as a safety
# net even if a user mistakenly allowlists them. The user-facing blocklist in
# Settings is the primary mechanism; this is a hardcoded backstop for the most
# dangerous categories (password managers, banking/health) so a typo in the
# allowlist cannot leak a vault. Documented as defense-in-depth.
# SINGLE SOURCE OF TRUTH NOTE [H7]: this baked tuple, the Swift
# ``dangerousBundleSubstrings`` constant (capture-helper), and the committed
# ``capture-helper/Sources/CaptureHelper/dangerous_apps.json`` resource are kept
# in lockstep by a parity unit test (tests/unit/test_capture.py). Edit ALL THREE
# together. The list is a SUPERSET (union) of every vendor either side carried.
#
# This baked tuple is the runtime source for Python: it is always complete and
# carries no filesystem dependency (so an installed wheel without the Swift
# sources still has the full backstop). The JSON file exists so the Swift helper
# can read one canonical list at build time and so the parity test can detect
# drift across all three copies. ``_DANGEROUS_BUNDLE_SUBSTRINGS`` is never empty
# or partial — the backstop fails closed.
_DANGEROUS_BUNDLE_SUBSTRINGS: tuple[str, ...] = (
    "1password",
    "onepassword",
    "lastpass",
    "bitwarden",
    "dashlane",
    "keepass",
    "keychain",
    "keychainaccess",
    "keeper",
    "nordpass",
    "enpass",
    "protonpass",
)


@dataclass(frozen=True)
class RedactionDecision:
    """The outcome of evaluating a capture event against the privacy policy.

    Attributes:
        capture: Whether the event's text may be ingested at all.
        reason: A short machine-readable reason code (e.g. ``"allowlisted"``,
            ``"not_allowlisted"``, ``"blocklisted"``, ``"incognito"``,
            ``"dangerous_app"``, ``"no_text"``). Never contains captured text.
        matched_rules: Names of any secret-scrubbing rules that fired on the
            text (populated only when ``capture`` is True and ``scrub`` ran).
    """

    capture: bool
    reason: str
    matched_rules: tuple[str, ...] = field(default_factory=tuple)


def _substring_any(haystack: str | None, needles: Iterable[str]) -> bool:
    """Case-insensitive substring test; ``None`` haystack never matches.

    Used ONLY for free-text signals (window titles / incognito markers), never
    for the allow/block bundle-id gate (see :func:`_bundle_matches`).
    """
    if not haystack:
        return False
    low = haystack.lower()
    return any(n.lower() in low for n in needles)


def _bundle_matches(app: str | None, entry: str) -> bool:
    """Match a canonical bundle id against ONE allow/block entry.

    Matching is **exact by default** (case-insensitive) to preserve the
    allowlist-ONLY guarantee: an app whose id merely *contains* an allowlisted
    substring must NOT pass the gate (PLAN allowlist-only first run). Two
    explicit opt-in syntaxes broaden a single entry:

      * ``glob:<pattern>`` — shell-style glob, e.g. ``glob:com.acme.*``.
      * ``re:<pattern>`` — anchored regular expression, e.g. ``re:com\\.acme\\..+``.

    Anything without one of these prefixes is compared for full equality, so a
    bare ``com.apple.mail`` only matches ``com.apple.mail`` exactly.
    """
    if not app or not entry:
        return False
    app_l = app.lower()
    entry = entry.strip()
    if entry.startswith("glob:"):
        return fnmatch.fnmatch(app_l, entry[len("glob:"):].strip().lower())
    if entry.startswith("re:"):
        try:
            return re.fullmatch(entry[len("re:"):].strip(), app, re.IGNORECASE) is not None
        except re.error:
            # A malformed user regex must fail closed (never silently match).
            return False
    return app_l == entry.lower()


def _bundle_matches_any(app: str | None, entries: Iterable[str]) -> bool:
    """True if ``app`` matches ANY allow/block entry via :func:`_bundle_matches`."""
    if not app:
        return False
    return any(_bundle_matches(app, e) for e in entries)


def is_incognito(app: str | None, window: str | None) -> bool:
    """Return True if app/window signals a private/incognito browsing context."""
    return _substring_any(window, _INCOGNITO_MARKERS) or _substring_any(
        app, _INCOGNITO_MARKERS
    )


def _is_blocklisted(app: str | None, blocklist) -> bool:
    """True if ``app`` matches a user blocklist entry (exact bundle id by default)."""
    if not app:
        return False
    return _bundle_matches_any(app, blocklist)


def _is_dangerous(app: str | None) -> bool:
    """True if ``app``'s bundle id is in a hardcoded dangerous category (backstop).

    The dangerous backstop is intentionally **substring-based**: it is a
    safety net for whole vendor families (e.g. any ``*1password*`` bundle id)
    and may only ever *reject*, so over-matching here is fail-safe.
    """
    return _substring_any(app, _DANGEROUS_BUNDLE_SUBSTRINGS)


def _is_allowlisted(app: str | None, allowlist) -> bool:
    """True if ``app`` matches an allowlist entry (exact bundle id by default).

    An empty allowlist means *nothing* is allowed (allowlist-only first run).
    Matching is exact unless an entry uses the explicit ``glob:``/``re:`` syntax
    (see :func:`_bundle_matches`).
    """
    if not allowlist:
        return False
    if not app:
        return False
    return _bundle_matches_any(app, allowlist)


def decide(
    *,
    app: str | None,
    window: str | None,
    text: str | None,
    incognito: bool = False,
    settings: Settings | None = None,
) -> RedactionDecision:
    """Decide whether a capture event may be ingested (allowlist-first).

    Order of evaluation (each can only *reject*, except the allowlist which is
    the sole enabler):

    1. There must be some text to capture.
    2. The app must be on the allowlist (empty allowlist => capture nothing).
    3. The app must not be on the user blocklist.
    4. The app must not be in the hardcoded dangerous-category backstop.
    5. The context must not be incognito/private (explicit flag or title heuristic).

    Args:
        app: App bundle id / name reported by the capture helper.
        window: Active window title.
        text: The captured text (only its presence is examined here).
        incognito: Explicit private-window signal from the helper, if known.
        settings: Settings providing allow/blocklists; defaults to
            :func:`get_settings`.

    Returns:
        A :class:`RedactionDecision`. ``reason`` is metadata-only and never
        contains captured text, satisfying the no-plaintext-in-logs rule [R4].
    """
    settings = settings or get_settings()

    if not text or not text.strip():
        return RedactionDecision(capture=False, reason="no_text")

    if not _is_allowlisted(app, settings.allowlist):
        return RedactionDecision(capture=False, reason="not_allowlisted")

    if _is_blocklisted(app, settings.blocklist):
        return RedactionDecision(capture=False, reason="blocklisted")

    if _is_dangerous(app):
        return RedactionDecision(capture=False, reason="dangerous_app")

    if incognito or is_incognito(app, window):
        return RedactionDecision(capture=False, reason="incognito")

    return RedactionDecision(capture=True, reason="allowlisted")


# ---------------------------------------------------------------------------
# Credit-card (PAN) redaction [H5]. A plain ``{13,16}`` regex both MISSED 17-19
# digit PANs and OVER-redacted long numeric IDs / order numbers. We instead:
#   1. Find candidate digit sequences (13-19 digits, optional SINGLE space/dash
#      group separators) bounded by ASCII non-[A-Za-z0-9_-] on both sides, so a
#      PAN embedded inside a longer alnum token is not torn out mid-stream.
#   2. Validate with the Luhn (mod-10) checksum — this drops false positives
#      (random/phone-like runs) and admits genuine 13-19 digit PANs.
#   3. Mask ONLY on whole-group boundaries:
#        * a CONTINUOUS run (no separators): mask iff the whole run is a valid
#          13-19 digit PAN; we NEVER scan arbitrary inner windows (a random
#          18/19-digit ID contains a Luhn-valid sub-window ~90% of the time, so
#          inner scanning would mass-false-positive legitimate IDs).
#        * a SEPARATED candidate (e.g. ``4111 1111 1111 1111 999``): mask the
#          longest PREFIX of whole groups that forms a valid 13-19 digit PAN,
#          leaving any trailing CVV/amount group intact. Group boundaries only —
#          never a mid-group split — so no dangling number fragment is produced.
# ---------------------------------------------------------------------------

_CARD_CANDIDATE_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:\d[ -]?){12,18}\d(?![A-Za-z0-9_-])")


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn (mod-10) checksum over a pure-digit string.

    ``digits`` must already be stripped of separators and be all-ASCII digits.
    Trivial values such as ``0000000000000000`` DO satisfy Luhn and will be
    masked — that is intentional and harmless for a defense-in-depth backstop.
    """
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48  # '0' == 48; caller guarantees ASCII digits
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _scrub_cards(text: str) -> tuple[str, bool]:
    """Mask Luhn-valid 13-19 digit PANs (group-anchored). Returns (text, masked?).

    See the module section comment above for the exact masking rules. Masking is
    always aligned to separator-group boundaries, never to arbitrary inner digit
    offsets, so legitimate long numeric identifiers are preserved.
    """
    masked_any = False
    out: list[str] = []
    last = 0
    for m in _CARD_CANDIDATE_RE.finditer(text):
        cand = m.group(0)
        # Positions (within ``cand``) of each digit, and the contiguous-digit
        # GROUPS delimited by the single separators that are actually present.
        digit_pos: list[int] = []
        groups: list[tuple[int, int]] = []  # (start_idx_in_digit_pos, end_exclusive)
        group_start = None
        prev_was_digit = False
        for idx, ch in enumerate(cand):
            if ch.isdigit():
                if not prev_was_digit:
                    group_start = len(digit_pos)
                digit_pos.append(idx)
                prev_was_digit = True
            else:  # a single space/dash separator between groups
                if prev_was_digit:
                    groups.append((group_start, len(digit_pos)))
                prev_was_digit = False
        if prev_was_digit:
            groups.append((group_start, len(digit_pos)))

        digits = "".join(cand[p] for p in digit_pos)
        has_separators = len(groups) > 1

        mask_end_digit: int | None = None  # exclusive index into digit_pos to mask up to
        if not has_separators:
            # Continuous run: only the whole run, never an inner window.
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                mask_end_digit = len(digit_pos)
        else:
            # Longest prefix of WHOLE groups forming a valid 13-19 digit PAN.
            for gi in range(len(groups), 0, -1):
                end_digit = groups[gi - 1][1]
                prefix = digits[:end_digit]
                if 13 <= len(prefix) <= 19 and _luhn_valid(prefix):
                    mask_end_digit = end_digit
                    break

        if mask_end_digit is None:
            continue

        # Map the digit span [0, mask_end_digit) back to a text slice of ``cand``:
        # from the first digit through the LAST masked digit (inclusive), which
        # includes interior separators but excludes any leading/trailing one.
        slice_start = digit_pos[0]
        slice_end = digit_pos[mask_end_digit - 1] + 1
        out.append(text[last : m.start()])
        out.append(cand[:slice_start])
        out.append("[REDACTED:card]")
        out.append(cand[slice_end:])
        last = m.end()
        masked_any = True

    out.append(text[last:])
    return "".join(out), masked_any


def scrub(text: str) -> tuple[str, tuple[str, ...]]:
    """Mask obvious secrets in captured text (defense-in-depth, best-effort).

    Args:
        text: Captured text that already passed :func:`decide`.

    Returns:
        ``(scrubbed_text, matched_rule_names)``. ``matched_rule_names`` records
        which patterns fired (metadata only — it never includes the secret
        values themselves), useful for diagnostics without leaking content.
    """
    matched: list[str] = []
    out = text
    for name, pattern, replacement in _SECRET_PATTERNS:
        new_out, n = pattern.subn(replacement, out)
        if n:
            matched.append(name)
            out = new_out
    # Cards are validated with Luhn, so they cannot live in the simple subn loop
    # above; run them last and only flag ``card_number`` when a PAN was masked.
    out, card_masked = _scrub_cards(out)
    if card_masked:
        matched.append("card_number")
    return out, tuple(matched)


# ---------------------------------------------------------------------------
# Metadata scrubbing [R4 fix]: window titles and URLs are stored alongside the
# text and frequently leak secrets too — URLs embed OAuth codes/access tokens,
# session ids, emails, document ids, or PHI in their query/fragment; window
# titles can carry full message/document content. We therefore scrub metadata
# separately BEFORE storage, not just the text body.
# ---------------------------------------------------------------------------

# Query-string parameter names whose VALUES are stripped even when query
# preservation is requested. Matched case-insensitively against the full key.
_SENSITIVE_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "access_token",
        "id_token",
        "refresh_token",
        "token",
        "code",
        "auth",
        "authorization",
        "api_key",
        "apikey",
        "key",
        "secret",
        "client_secret",
        "password",
        "passwd",
        "pwd",
        "session",
        "sessionid",
        "sid",
        "sig",
        "signature",
        "email",
        "e",
        "otp",
        "state",
        "assertion",
        "ticket",
    }
)


def scrub_url(url: str | None, *, keep_query: bool = False) -> str | None:
    """Scrub a captured URL before storage (best-effort, defense-in-depth).

    By default the **entire query string and fragment are dropped**, since they
    routinely carry OAuth codes, access tokens, session ids, emails, and
    document ids that have no place in a memory store. Only the scheme, host,
    and path survive.

    Args:
        url: The raw URL reported by the helper (may be ``None``/empty).
        keep_query: If True, preserve the query but still **redact the values of
            sensitive keys** (see :data:`_SENSITIVE_QUERY_KEYS`); the fragment is
            always dropped (fragments commonly hold implicit-flow access tokens).

    Returns:
        The scrubbed URL, or the input unchanged when it cannot be parsed as a
        hierarchical URL (so we never silently corrupt a non-URL title).
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.scheme and not parts.netloc:
        # Not a hierarchical URL (e.g. a bare title) — leave it to the caller's
        # title scrubbing rather than mangling it here.
        return url

    query = ""
    if keep_query and parts.query:
        kept: list[tuple[str, str]] = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            if k.lower() in _SENSITIVE_QUERY_KEYS:
                kept.append((k, "[REDACTED]"))
            else:
                kept.append((k, v))
        query = urlencode(kept)

    # Fragment is always discarded.
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def scrub_title(title: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Run a window title through the same secret patterns as body text.

    Window titles can expose full message bodies, document contents, or pasted
    secrets. Returns ``(scrubbed_title_or_None, matched_rule_names)``; ``None``
    input passes through as ``(None, ())``.
    """
    if title is None:
        return None, ()
    return scrub(title)


def scrub_metadata(
    *,
    window: str | None,
    url: str | None,
    keep_query: bool = False,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Scrub the non-text metadata fields (window title + URL) before storage.

    Returns ``(scrubbed_window, scrubbed_url, matched_rule_names)`` where the
    matched rules come from secret patterns that fired on the window title
    (metadata only — never the values themselves).
    """
    scrubbed_window, matched = scrub_title(window)
    scrubbed_url = scrub_url(url, keep_query=keep_query)
    return scrubbed_window, scrubbed_url, matched


def apply(
    *,
    app: str | None,
    window: str | None,
    text: str | None,
    incognito: bool = False,
    settings: Settings | None = None,
) -> tuple[RedactionDecision, str | None]:
    """Run the full policy: decide, then scrub if accepted.

    Returns ``(decision, scrubbed_text_or_None)``. When the decision rejects the
    event, the second element is ``None`` and no text is returned (so callers
    cannot accidentally ingest rejected content). When accepted, the returned
    text is the scrubbed version and ``decision.matched_rules`` lists the rules
    that fired.
    """
    decision = decide(
        app=app, window=window, text=text, incognito=incognito, settings=settings
    )
    if not decision.capture:
        return decision, None

    assert text is not None  # guaranteed by decide() "no_text" path
    scrubbed, matched = scrub(text)
    decision = RedactionDecision(
        capture=True, reason=decision.reason, matched_rules=matched
    )
    return decision, scrubbed


__all__ = [
    "RedactionDecision",
    "decide",
    "scrub",
    "scrub_url",
    "scrub_title",
    "scrub_metadata",
    "apply",
    "is_incognito",
]
