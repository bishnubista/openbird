"""Deterministic daily memory distillation.

This module turns raw capture observations into a compact, descriptive day
artifact before any LLM sees the data. It intentionally stores facts and
heuristics only: time/category/session metrics, source ids, and extracted
domains/repos/title tokens. Coaching language and productivity judgments are
read-time inferences for later features, not durable facts.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse

from openbird.types import Observation

EXTRACTOR_VERSION = "day-memory-v1"

_BROWSER_APPS = {
    "com.google.chrome",
    "com.apple.safari",
    "org.mozilla.firefox",
    "company.thebrowser.browser",
    "com.brave.browser",
    "com.microsoft.edgemac",
}
_CODING_HINTS = (
    "ghostty", "terminal", "iterm", "kitty", "warp", "wezterm", "code",
    "xcode", "jetbrains", "pycharm", "intellij", "cursor", "codex", "claude",
)
_COMM_HINTS = (
    "slack", "discord", "messages", "mobilesms", "mail", "gmail", "linkedin",
    "zoom", "meet", "teams",
)
_NOTES_HINTS = (
    "notes", "notion", "obsidian", "preview", "pages", "word", "docs",
    "pdf", "arxiv",
)
_SYSTEM_HINTS = ("systempreferences", "settings", "keychain", "activitymonitor")
_FILE_HINTS = ("finder",)
_MEDIA_DOMAINS = ("youtube.com", "youtu.be", "netflix.com", "spotify.com")
_REPO_RE = re.compile(r"\bgithub\.com/([^/\s]+)/([^/\s?#]+)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
_STOP_TOKENS = {
    "google", "chrome", "safari", "github", "pull", "request", "requests",
    "openbird", "codex", "page", "copy", "login", "sign", "into", "with",
    "http", "https", "www", "com",
}


@dataclass(frozen=True)
class DayMemoryBuild:
    """Built payload plus the source ids it depends on."""

    payload: dict
    source_ids: list[str]


def local_date_for_window(start_ts: float) -> str:
    """Return the local YYYY-MM-DD date for a day window start."""
    return dt.datetime.fromtimestamp(start_ts).date().isoformat()


def build_day_memory(
    rows: list[tuple[Observation, str]],
    *,
    start_ts: float,
    end_ts: float,
    day_offset: int,
    source_scope: str = "capture",
    gap_seconds: float = 300.0,
) -> DayMemoryBuild:
    """Build a deterministic, no-model day-memory payload."""
    ordered = sorted(rows, key=lambda item: item[0].ts)
    source_ids = [obs.id for obs, _ in ordered]
    sessions = _build_sessions(ordered)
    timed = _timed_observations(ordered, gap_seconds=gap_seconds)

    time_by_hour: Counter[str] = Counter()
    time_by_app: Counter[str] = Counter()
    time_by_category: Counter[str] = Counter()
    for obs, _text, category, seconds in timed:
        if seconds <= 0:
            continue
        hour = dt.datetime.fromtimestamp(obs.ts).strftime("%H:00")
        time_by_hour[hour] += seconds
        time_by_app[obs.app or "unknown"] += seconds
        time_by_category[category] += seconds

    payload = {
        "schema": 1,
        "extractor_version": EXTRACTOR_VERSION,
        "narrative_status": "not_persisted",
        "local_date": local_date_for_window(start_ts),
        "source_scope": source_scope,
        "day_offset": day_offset,
        "window": {"start": start_ts, "end": end_ts},
        "coverage": {
            "observations": len(ordered),
            "sessions": len(sessions),
            "apps": len({obs.app for obs, _ in ordered if obs.app}),
            "source_ids": source_ids,
        },
        "sessions": sessions,
        "metrics": {
            "active_seconds": round(sum(time_by_category.values()), 3),
            "time_by_hour": _round_counter(time_by_hour),
            "time_by_app": _round_counter(time_by_app),
            "time_by_category": _round_counter(time_by_category),
            "context_switch_count": _context_switch_count(ordered),
            "longest_same_category_streak": _longest_category_streak(timed),
            "first_seen": ordered[0][0].ts if ordered else None,
            "last_seen": ordered[-1][0].ts if ordered else None,
            "unknown_category_count": sum(
                1 for obs, text in ordered if classify_observation(obs, text)[0] == "unknown"
            ),
        },
        "entities": _entities(ordered),
    }
    return DayMemoryBuild(payload=payload, source_ids=source_ids)


def classify_observation(obs: Observation, text: str = "") -> tuple[str, float]:
    """Classify one observation into a descriptive activity category."""
    hay = " ".join(
        part for part in (obs.app or "", obs.window or "", obs.url or "", text[:500]) if part
    ).lower()
    app = (obs.app or "").lower()
    domain = _domain(obs.url or "")

    if domain and any(domain.endswith(d) for d in _MEDIA_DOMAINS):
        return "browser_media", 0.9
    if any(h in hay for h in _COMM_HINTS):
        return "communication", 0.75
    if any(h in hay for h in _SYSTEM_HINTS):
        return "system_admin", 0.8
    if any(h in app for h in _FILE_HINTS):
        return "file_management", 0.8
    if any(h in hay for h in _CODING_HINTS):
        return "coding", 0.75
    if any(h in hay for h in _NOTES_HINTS):
        return "notes_docs", 0.7
    if app in _BROWSER_APPS or domain:
        return "browser_research", 0.65
    return "unknown", 0.2


def _build_sessions(rows: list[tuple[Observation, str]]) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[Observation, str]]] = defaultdict(list)
    for obs, text in rows:
        key = (obs.session_id or obs.id, obs.app or "unknown")
        grouped[key].append((obs, text))

    sessions: list[dict] = []
    for (_bucket, app), items in grouped.items():
        items.sort(key=lambda item: item[0].ts)
        category, confidence = _classify_many(items)
        title = _representative_title(items)
        source_ids = [obs.id for obs, _ in items]
        sessions.append(
            {
                "app": app,
                "category": category,
                "category_confidence": confidence,
                "start": items[0][0].ts,
                "end": items[-1][0].ts,
                "count": len(items),
                "title": title,
                "source_ids": source_ids,
            }
        )
    sessions.sort(key=lambda s: s["start"])
    return sessions


def _classify_many(items: list[tuple[Observation, str]]) -> tuple[str, float]:
    votes: Counter[str] = Counter()
    confidences: dict[str, list[float]] = defaultdict(list)
    for obs, text in items:
        category, confidence = classify_observation(obs, text)
        votes[category] += 1
        confidences[category].append(confidence)
    category, _ = votes.most_common(1)[0]
    values = confidences[category]
    return category, round(sum(values) / len(values), 3)


def _representative_title(items: list[tuple[Observation, str]]) -> str | None:
    titles = [obs.window or obs.url for obs, _ in items if obs.window or obs.url]
    if not titles:
        return None
    counts = Counter(titles)
    return max(titles, key=lambda title: (counts[title], _latest_title_ts(items, title)))


def _latest_title_ts(items: list[tuple[Observation, str]], title: str) -> float:
    return max(obs.ts for obs, _ in items if (obs.window or obs.url) == title)


def _timed_observations(
    rows: list[tuple[Observation, str]], *, gap_seconds: float
) -> list[tuple[Observation, str, str, float]]:
    out: list[tuple[Observation, str, str, float]] = []
    for idx, (obs, text) in enumerate(rows):
        category, _confidence = classify_observation(obs, text)
        if idx + 1 >= len(rows):
            seconds = 0.0
        else:
            seconds = max(0.0, min(rows[idx + 1][0].ts - obs.ts, gap_seconds))
        out.append((obs, text, category, seconds))
    return out


def _context_switch_count(rows: list[tuple[Observation, str]]) -> int:
    if not rows:
        return 0
    count = 0
    prev = (rows[0][0].app, classify_observation(rows[0][0], rows[0][1])[0])
    for obs, text in rows[1:]:
        cur = (obs.app, classify_observation(obs, text)[0])
        if cur != prev:
            count += 1
        prev = cur
    return count


def _longest_category_streak(
    timed: list[tuple[Observation, str, str, float]]
) -> dict | None:
    best_category: str | None = None
    best_seconds = 0.0
    cur_category: str | None = None
    cur_seconds = 0.0
    for _obs, _text, category, seconds in timed:
        if category != cur_category:
            if cur_category is not None and cur_seconds > best_seconds:
                best_category, best_seconds = cur_category, cur_seconds
            cur_category, cur_seconds = category, seconds
        else:
            cur_seconds += seconds
    if cur_category is not None and cur_seconds > best_seconds:
        best_category, best_seconds = cur_category, cur_seconds
    if best_category is None:
        return None
    return {"category": best_category, "seconds": round(best_seconds, 3)}


def _entities(rows: list[tuple[Observation, str]]) -> dict:
    domains: dict[str, set[str]] = defaultdict(set)
    repos: dict[str, set[str]] = defaultdict(set)
    tokens: Counter[str] = Counter()
    token_sources: dict[str, set[str]] = defaultdict(set)

    for obs, text in rows:
        blob = " ".join(part for part in (obs.window or "", obs.url or "", text[:500]) if part)
        domain = _domain(obs.url or "") or _domain(blob)
        if domain:
            domains[domain].add(obs.id)
        for owner, repo in _REPO_RE.findall(blob):
            clean = f"{owner}/{repo}".rstrip("/")
            repos[clean].add(obs.id)
        for token in _TOKEN_RE.findall(blob):
            lowered = token.lower().strip("._-")
            if len(lowered) < 3 or lowered in _STOP_TOKENS or lowered.isdigit():
                continue
            tokens[lowered] += 1
            token_sources[lowered].add(obs.id)

    return {
        "domains": _rank_entity_map(domains),
        "repos": _rank_entity_map(repos),
        "title_tokens": [
            {"token": token, "count": count, "source_ids": sorted(token_sources[token])[:5]}
            for token, count in tokens.most_common(12)
        ],
    }


def _domain(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    if "://" not in text:
        match = re.search(r"\b([a-z0-9.-]+\.[a-z]{2,})(?:/[^\s]*)?", text, re.I)
        if not match:
            return None
        text = "https://" + match.group(0)
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _rank_entity_map(items: dict[str, set[str]]) -> list[dict]:
    ranked = sorted(items.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [
        {"value": value, "count": len(source_ids), "source_ids": sorted(source_ids)[:5]}
        for value, source_ids in ranked[:12]
    ]


def _round_counter(counter: Counter[str]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in sorted(counter.items())}

