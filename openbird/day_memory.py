"""Deterministic daily memory distillation.

This module turns raw capture observations into a compact, descriptive day
artifact before any LLM sees the data. It intentionally stores facts and
heuristics only: time/category/session metrics, source ids, and extracted
domains/repos/title tokens. Coaching language and productivity judgments are
read-time inferences for later features, not durable facts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse

from openbird.types import Observation

EXTRACTOR_VERSION = "day-memory-v2"

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
_GITHUB_ITEM_RE = re.compile(
    r"github\.com/([^/\s]+)/([^/\s?#]+)/(issues|pull)/(\d+)", re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
_OPEN_LOOP_RE = re.compile(
    r"\b(todo|follow(?:ing)? up|follow-up|blocked|blocker|review|fix|bug|issue|pr|pull request)\b",
    re.IGNORECASE,
)
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
    source_fingerprint: dict | None = None,
    as_of: float | None = None,
) -> DayMemoryBuild:
    """Build a deterministic, no-model day-memory payload."""
    ordered = sorted(rows, key=lambda item: _observation_sort_key(item[0]))
    source_ids = [obs.id for obs, _ in ordered]
    sessions = _build_sessions(ordered)
    timed = _timed_observations(ordered, gap_seconds=gap_seconds, end_ts=end_ts)
    active_by_source = _active_seconds_by_source(timed)
    entities = _entities(ordered)
    workstreams = _workstreams(sessions, entities, active_by_source)
    open_loops = _open_loops(ordered)

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
        "as_of": end_ts if as_of is None else as_of,
        "day_offset": day_offset,
        "window": {"start": start_ts, "end": end_ts},
        "coverage": {
            "observations": len(ordered),
            "sessions": len(sessions),
            "apps": len({obs.app for obs, _ in ordered if obs.app}),
            "source_ids": source_ids,
        },
        "source_fingerprint": source_fingerprint or source_fingerprint_for_rows(ordered),
        "sessions": sessions,
        "workstreams": workstreams,
        "open_loops": open_loops,
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
        "entities": entities,
    }
    return DayMemoryBuild(payload=payload, source_ids=source_ids)


def source_fingerprint_for_rows(rows: list[tuple[Observation, str]]) -> dict:
    """Return a stable fingerprint for the exact observation membership."""
    ordered = sorted(rows, key=lambda item: item[0].id)
    pairs = [(obs.id, obs.content_hash) for obs, _text in ordered]
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    timestamps = [obs.ts for obs, _text in ordered]
    return {
        "count": len(ordered),
        "min_ts": min(timestamps) if timestamps else None,
        "max_ts": max(timestamps) if timestamps else None,
        "ids_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


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
    for (_bucket, app), items in sorted(grouped.items(), key=lambda item: item[0]):
        items.sort(key=lambda item: _observation_sort_key(item[0]))
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
    sessions.sort(
        key=lambda s: (
            s["start"],
            s["end"],
            s["app"],
            s["category"],
            tuple(s["source_ids"]),
        )
    )
    return sessions


def _classify_many(items: list[tuple[Observation, str]]) -> tuple[str, float]:
    votes: Counter[str] = Counter()
    confidences: dict[str, list[float]] = defaultdict(list)
    for obs, text in items:
        category, confidence = classify_observation(obs, text)
        votes[category] += 1
        confidences[category].append(confidence)
    category, _ = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0]
    values = confidences[category]
    return category, round(sum(values) / len(values), 3)


def _representative_title(items: list[tuple[Observation, str]]) -> str | None:
    titles = [obs.window or obs.url for obs, _ in items if obs.window or obs.url]
    if not titles:
        return None
    counts = Counter(titles)
    return sorted(
        counts,
        key=lambda title: (-counts[title], -_latest_title_ts(items, title), title),
    )[0]


def _latest_title_ts(items: list[tuple[Observation, str]], title: str) -> float:
    return max(obs.ts for obs, _ in items if (obs.window or obs.url) == title)


def _timed_observations(
    rows: list[tuple[Observation, str]], *, gap_seconds: float, end_ts: float
) -> list[tuple[Observation, str, str, float]]:
    out: list[tuple[Observation, str, str, float]] = []
    for idx, (obs, text) in enumerate(rows):
        category, _confidence = classify_observation(obs, text)
        if idx + 1 >= len(rows):
            seconds = max(0.0, min(end_ts - obs.ts, gap_seconds))
        else:
            seconds = max(0.0, min(rows[idx + 1][0].ts - obs.ts, gap_seconds))
        out.append((obs, text, category, seconds))
    return out


def _active_seconds_by_source(
    timed: list[tuple[Observation, str, str, float]]
) -> dict[str, float]:
    seconds: Counter[str] = Counter()
    for obs, _text, _category, duration in timed:
        if duration > 0:
            seconds[obs.id] += duration
    return {source_id: round(value, 3) for source_id, value in sorted(seconds.items())}


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
            for token, count in sorted(tokens.items(), key=lambda item: (-item[1], item[0]))[
                :12
            ]
        ],
    }


def _workstreams(
    sessions: list[dict], entities: dict, active_by_source: dict[str, float]
) -> list[dict]:
    session_by_source: dict[str, dict] = {}
    category_by_source: dict[str, str] = {}
    for session in sessions:
        for source_id in session.get("source_ids", []):
            session_by_source[source_id] = session
            category_by_source[source_id] = session.get("category", "unknown")

    candidates: list[dict] = []
    for kind in ("repo", "domain"):
        entity_key = "repos" if kind == "repo" else "domains"
        for item in entities.get(entity_key, []):
            source_ids = list(item.get("source_ids", []))
            if not source_ids:
                continue
            categories = Counter(category_by_source.get(sid, "unknown") for sid in source_ids)
            category = sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            sessions_seen = {
                tuple(session_by_source[sid].get("source_ids", []))
                for sid in source_ids
                if sid in session_by_source
            }
            candidates.append(
                {
                    "kind": kind,
                    "label": item["value"],
                    "category": category,
                    "session_count": len(sessions_seen),
                    "active_seconds": round(sum(active_by_source.get(sid, 0.0) for sid in source_ids), 3),
                    "source_ids": sorted(source_ids),
                    "source_count": item.get("count", len(source_ids)),
                }
            )

    by_category: dict[str, set[str]] = defaultdict(set)
    for session in sessions:
        by_category[session.get("category", "unknown")].update(session.get("source_ids", []))
    for category, ids in sorted(by_category.items()):
        source_ids = sorted(ids)
        candidates.append(
            {
                "kind": "category",
                "label": category,
                "category": category,
                "session_count": sum(
                    1 for session in sessions if session.get("category") == category
                ),
                "active_seconds": round(sum(active_by_source.get(sid, 0.0) for sid in source_ids), 3),
                "source_ids": source_ids[:12],
                "source_count": len(source_ids),
            }
        )

    return sorted(
        candidates,
        key=lambda item: (
            -float(item.get("active_seconds", 0.0)),
            -int(item.get("source_count", 0)),
            item.get("kind", ""),
            item.get("label", ""),
        ),
    )[:12]


def _open_loops(rows: list[tuple[Observation, str]]) -> list[dict]:
    loops: dict[tuple[str, str], dict] = {}
    for obs, text in rows:
        blob = " ".join(part for part in (obs.window or "", obs.url or "", text[:500]) if part)
        if not blob:
            continue
        github = _GITHUB_ITEM_RE.search(blob)
        cue_match = _OPEN_LOOP_RE.search(blob)
        if not github and not cue_match:
            continue
        if github:
            owner, repo, item_kind, number = github.groups()
            kind = "github_pr" if item_kind == "pull" else "github_issue"
            cue = f"{owner}/{repo} {item_kind} #{number}"
            key_value = cue.casefold()
        else:
            kind = "cue"
            cue = cue_match.group(0).lower() if cue_match else "cue"
            key_value = None
        title = _clean_title(obs.window or text or obs.url or cue)
        key = (kind, key_value or title.casefold())
        current = loops.setdefault(
            key,
            {
                "kind": kind,
                "title": title,
                "cue": cue,
                "source_ids": [],
                "source_count": 0,
            },
        )
        current["source_count"] += 1
        current["source_ids"].append(obs.id)

    out: list[dict] = []
    for item in loops.values():
        item["source_ids"] = sorted(set(item["source_ids"]))[:12]
        out.append(item)
    return sorted(
        out,
        key=lambda item: (-int(item["source_count"]), item["kind"], item["title"]),
    )[:12]


def _clean_title(value: str, *, limit: int = 140) -> str:
    title = " ".join(value.split())
    if len(title) > limit:
        title = title[: limit - 1].rstrip() + "…"
    return title


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


def _observation_sort_key(obs: Observation) -> tuple[float, str, str, str, str]:
    return (obs.ts, obs.id, obs.app or "", obs.window or "", obs.url or "")
