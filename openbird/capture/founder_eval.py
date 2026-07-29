"""Privacy-safe, no-model capture-quality evaluation for founder recaps."""

from __future__ import annotations

import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from openbird.capture.attempts import (
    CAPTURE_ADAPTERS,
    CAPTURE_COMPLETENESS,
    CAPTURE_OUTCOMES,
    CAPTURE_REASON_CODES,
    CAPTURE_TRIGGERS,
)
from openbird.capture.health import build_capture_health

SNAPSHOT_VERSION = 1
SNAPSHOT_NAME = "founder-context-eval.json"
RECENT_WINDOW_SECONDS = 5 * 86_400.0
FRESHNESS_MAX_AGE_SECONDS = 36 * 3_600.0
MAX_SNAPSHOT_APP_IDS = 100
MAX_SNAPSHOT_BYTES = 128 * 1024
_SOURCES = ("capture", "meeting", "ingest", "mcp")


def snapshot_path(settings: Any) -> Path:
    return Path(settings.data_dir) / "logs" / SNAPSHOT_NAME


def read_snapshot(path: Path) -> dict[str, Any] | None:
    """Read a prior snapshot for deltas; malformed files are ignored."""
    try:
        if path.stat().st_size > MAX_SNAPSHOT_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_snapshot(report: dict[str, Any], path: Path) -> None:
    """Atomically replace one bounded, owner-only JSON snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    # The schema contains bounded lists and aggregate counters only. Retain a
    # defensive ceiling so an accidental future unbounded field cannot turn the
    # scheduled evaluator into a disk-growth surface.
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise ValueError("founder-context eval snapshot exceeded size bound")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def invalidate_snapshot(*, settings: Any | None = None, data_dir: Path | None = None) -> None:
    """Remove derived metadata before source deletion/retention commits.

    A missing snapshot is already the desired state. Every other filesystem
    error intentionally propagates so the caller can roll back its source
    deletion instead of leaving stale app/source metadata behind.
    """
    if data_dir is None:
        if settings is None:
            from openbird.config import get_settings

            settings = get_settings()
        data_dir = Path(settings.data_dir)
    target = data_dir / "logs" / SNAPSHOT_NAME
    target.unlink(missing_ok=True)


def _one_value(row: Any) -> int:
    if row is None:
        return 0
    value = next(iter(row.values())) if isinstance(row, dict) else row[0]
    return int(value or 0)


def _file_sizes(settings: Any) -> dict[str, int]:
    db = Path(str(settings.db_path))

    def size(path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    return {
        "db_bytes": size(db),
        "wal_bytes": size(Path(f"{db}-wal")),
        "shm_bytes": size(Path(f"{db}-shm")),
    }


def _storage_metrics(store: Any, settings: Any) -> dict[str, int]:
    values = _file_sizes(settings)
    for pragma in ("page_count", "page_size", "freelist_count"):
        values[pragma] = _one_value(store.conn.execute(f"PRAGMA {pragma}").fetchone())
    values["reclaimable_bytes"] = (
        values["page_size"] * values["freelist_count"]
    )
    return values


def _previous_value(previous: dict[str, Any] | None, section: str, key: str) -> int:
    if not previous:
        return 0
    value = previous.get(section, {})
    if not isinstance(value, dict):
        return 0
    try:
        return int(value.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _peak_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux/BSD commonly report KiB.
    return rss if sys.platform == "darwin" else rss * 1024


def _corpus_metrics(store: Any, start_ts: float, end_ts: float) -> dict[str, Any]:
    row = store.conn.execute(
        """
        SELECT
            COUNT(*) AS observations,
            COUNT(DISTINCT o.content_hash) AS distinct_contexts,
            COUNT(DISTINCT COALESCE(o.session_id, o.id)) AS sessions,
            COUNT(DISTINCT CASE WHEN o.app IS NOT NULL AND o.app != '' THEN o.app END)
                AS app_count,
            MAX(o.ts) AS last_observation_ts,
            COALESCE(SUM(LENGTH(CAST(COALESCE(b.text, '') AS BLOB))), 0)
                AS recent_text_bytes,
            SUM(CASE WHEN LENGTH(COALESCE(b.text, '')) >= 120 THEN 1 ELSE 0 END)
                AS substantive_observations,
            SUM(CASE WHEN
                LOWER(b.text) LIKE '%decided%' OR
                LOWER(b.text) LIKE '%decision%' OR
                LOWER(b.text) LIKE '%chose%' OR
                LOWER(b.text) LIKE '%agreed%'
                THEN 1 ELSE 0 END) AS decision_cues,
            SUM(CASE WHEN
                LOWER(b.text) LIKE '%shipped%' OR
                LOWER(b.text) LIKE '%merged%' OR
                LOWER(b.text) LIKE '%implemented%' OR
                LOWER(b.text) LIKE '%finished%' OR
                LOWER(b.text) LIKE '%fixed%'
                THEN 1 ELSE 0 END) AS progress_cues,
            SUM(CASE WHEN
                LOWER(b.text) LIKE '%todo%' OR
                LOWER(b.text) LIKE '%follow up%' OR
                LOWER(b.text) LIKE '%next step%' OR
                LOWER(b.text) LIKE '%blocked%' OR
                LOWER(b.text) LIKE '%still need%'
                THEN 1 ELSE 0 END) AS open_loop_cues
        FROM observations o
        JOIN content_blobs b ON b.content_hash = o.content_hash
        WHERE o.ts >= ? AND o.ts <= ?
          AND o.source IN ('capture','meeting','ingest','mcp')
        """,
        (float(start_ts), float(end_ts)),
    ).fetchone()
    raw = dict(row or {})
    observations = int(raw.get("observations") or 0)
    sources = {name: 0 for name in _SOURCES}
    for source_row in store.conn.execute(
        """
        SELECT source, COUNT(*) AS count
        FROM observations
        WHERE ts >= ? AND ts <= ?
          AND source IN ('capture','meeting','ingest','mcp')
        GROUP BY source
        """,
        (float(start_ts), float(end_ts)),
    ).fetchall():
        sources[str(source_row["source"])] = int(source_row["count"] or 0)
    apps = [
        str(item["app"])
        for item in store.conn.execute(
            """
            SELECT app, COUNT(*) AS count
            FROM observations
            WHERE ts >= ? AND ts <= ?
              AND source IN ('capture','meeting','ingest','mcp')
              AND app IS NOT NULL AND app != ''
            GROUP BY app
            ORDER BY count DESC, app
            LIMIT ?
            """,
            (float(start_ts), float(end_ts), MAX_SNAPSHOT_APP_IDS),
        ).fetchall()
    ]
    substantive = int(raw.get("substantive_observations") or 0)
    return {
        "window_seconds": RECENT_WINDOW_SECONDS,
        "observations": observations,
        "distinct_contexts": int(raw.get("distinct_contexts") or 0),
        "sessions": int(raw.get("sessions") or 0),
        "app_count": int(raw.get("app_count") or 0),
        "app_ids": apps,
        "app_ids_truncated": int(raw.get("app_count") or 0) > len(apps),
        "source_counts": sources,
        "last_observation_ts": raw.get("last_observation_ts"),
        "recent_text_bytes": int(raw.get("recent_text_bytes") or 0),
        "substantive_observations": substantive,
        "substantive_ratio": round(substantive / observations, 4)
        if observations
        else 0.0,
        "cue_counts": {
            "decision": int(raw.get("decision_cues") or 0),
            "progress": int(raw.get("progress_cues") or 0),
            "open_loop": int(raw.get("open_loop_cues") or 0),
        },
    }


def _attempt_metrics(store: Any, start_ts: float) -> dict[str, Any]:
    row = store.conn.execute(
        """
        SELECT
            COUNT(*) AS attempts,
            SUM(CASE WHEN status = 'finished' THEN 1 ELSE 0 END) AS finished,
            COALESCE(SUM(bytes_emitted), 0) AS bytes_emitted,
            COALESCE(AVG(CASE WHEN status = 'finished' THEN elapsed_ms END), 0)
                AS average_elapsed_ms,
            COALESCE(MAX(elapsed_ms), 0) AS max_elapsed_ms,
            SUM(CASE WHEN outcome = 'failed_bounded' THEN 1 ELSE 0 END)
                AS failed_bounded,
            SUM(CASE WHEN reason_codes_json LIKE '%"budget_exhausted"%'
                THEN 1 ELSE 0 END) AS budget_exhausted
        FROM capture_attempts
        WHERE trigger_ts >= ?
        """,
        (float(start_ts),),
    ).fetchone()
    raw = dict(row or {})
    reason_counts = {reason: 0 for reason in sorted(CAPTURE_REASON_CODES)}
    for reason_row in store.conn.execute(
        """
        SELECT j.value AS reason, COUNT(*) AS count
        FROM capture_attempts a, json_each(a.reason_codes_json) j
        WHERE a.trigger_ts >= ?
        GROUP BY j.value
        """,
        (float(start_ts),),
    ).fetchall():
        reason = str(reason_row["reason"])
        if reason in reason_counts:
            reason_counts[reason] = int(reason_row["count"] or 0)

    def closed_counts(column: str, allowed: set[str] | frozenset[str]) -> dict[str, int]:
        counts = {name: 0 for name in sorted(allowed)}
        # ``column`` is selected only from a module-owned closed tuple below.
        for grouped in store.conn.execute(
            f"SELECT {column} AS value, COUNT(*) AS count "
            "FROM capture_attempts WHERE trigger_ts >= ? "
            f"AND {column} IS NOT NULL GROUP BY {column}",
            (float(start_ts),),
        ).fetchall():
            value = str(grouped["value"])
            if value in counts:
                counts[value] = int(grouped["count"] or 0)
        return counts

    return {
        "attempts": int(raw.get("attempts") or 0),
        "finished": int(raw.get("finished") or 0),
        "bytes_emitted": int(raw.get("bytes_emitted") or 0),
        "average_elapsed_ms": round(float(raw.get("average_elapsed_ms") or 0.0), 2),
        "max_elapsed_ms": int(raw.get("max_elapsed_ms") or 0),
        "failed_bounded": int(raw.get("failed_bounded") or 0),
        "budget_exhausted": int(raw.get("budget_exhausted") or 0),
        "reason_counts": reason_counts,
        "trigger_counts": closed_counts("trigger", CAPTURE_TRIGGERS),
        "adapter_counts": closed_counts("adapter_id", CAPTURE_ADAPTERS),
        "completeness_counts": closed_counts(
            "completeness", CAPTURE_COMPLETENESS
        ),
        "outcome_counts": closed_counts("outcome", CAPTURE_OUTCOMES),
    }


def _assessment(
    corpus: dict[str, Any], daemon: dict[str, Any], *, now: float
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    observations = int(corpus["observations"])
    last_ts = corpus.get("last_observation_ts")
    if observations < 12:
        reasons.append("insufficient_recent_observations")
    if int(corpus["distinct_contexts"]) < 4:
        reasons.append("insufficient_distinct_contexts")
    if int(corpus["app_count"]) < 1:
        reasons.append("insufficient_app_coverage")
    if last_ts is None or now - float(last_ts) > FRESHNESS_MAX_AGE_SECONDS:
        reasons.append("capture_stale")
    hard = {
        "insufficient_recent_observations",
        "insufficient_distinct_contexts",
        "insufficient_app_coverage",
        "capture_stale",
    }
    if any(reason in hard for reason in reasons):
        return "not_ready", reasons

    if float(corpus["substantive_ratio"]) < 0.3:
        reasons.append("shallow_context")
    signaled = sum(int(count) > 0 for count in corpus["cue_counts"].values())
    if signaled < 2:
        reasons.append("missing_work_signals")
    if daemon.get("state") != "ok":
        reasons.append("daemon_not_healthy")
    return ("partial", reasons) if reasons else ("ready", [])


def evaluate_store(
    store: Any,
    *,
    settings: Any,
    now: float | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an already-open maintenance store using aggregate reads only."""
    started = time.perf_counter()
    generated_at = time.time() if now is None else float(now)
    start_ts = generated_at - RECENT_WINDOW_SECONDS
    corpus = _corpus_metrics(store, start_ts, generated_at)
    # Corpus SQL above already provides the bounded five-day app/source view.
    # Capture health needs only daemon/pause state here; avoid its otherwise
    # all-history per-app aggregation because the scheduled evaluator discards
    # those health rows.
    health = build_capture_health(
        settings=settings,
        activity_by_app={},
        generated_at=generated_at,
        recent_window_seconds=RECENT_WINDOW_SECONDS,
    )
    storage = _storage_metrics(store, settings)
    storage["db_bytes_delta"] = storage["db_bytes"] - _previous_value(
        previous, "storage", "db_bytes"
    )
    storage["recent_text_bytes_delta"] = corpus["recent_text_bytes"] - _previous_value(
        previous, "corpus", "recent_text_bytes"
    )
    attempts = _attempt_metrics(store, start_ts)
    state, reasons = _assessment(corpus, health["daemon"], now=generated_at)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": generated_at,
        "state": state,
        "reason_codes": reasons,
        "corpus": corpus,
        "capture": {
            "daemon": health["daemon"],
            "paused": bool(health["paused"]),
            "attempts": attempts,
        },
        "storage": storage,
        "resources": {
            "elapsed_ms": elapsed_ms,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def evaluate_and_record_snapshot(
    store: Any,
    *,
    settings: Any,
    path: Path,
    now: float | None = None,
) -> dict[str, Any]:
    """Evaluate and replace the snapshot under the store's writer lock.

    Observation deletion uses ``BEGIN IMMEDIATE`` too. Holding that same short
    lock from aggregate reads through the atomic file replacement gives the two
    operations a strict order:

    * evaluation first -> deletion waits, then removes the just-written snapshot;
    * deletion first -> evaluation waits, then reads only surviving rows.

    No model call belongs in this helper. The caller may add a manual probe to
    its transient output only after this transaction has committed.
    """
    store.conn.execute("BEGIN IMMEDIATE")
    try:
        report = evaluate_store(
            store,
            settings=settings,
            now=now,
            previous=read_snapshot(path),
        )
        write_snapshot(report, path)
        store.conn.commit()
    except Exception:
        store.conn.rollback()
        raise
    return report


def unavailable_report(
    *,
    settings: Any,
    reason: str,
    now: float | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a file/storage-only closed snapshot when the store cannot open."""
    generated_at = time.time() if now is None else float(now)
    storage = _file_sizes(settings)
    storage.update(
        {
            "page_count": 0,
            "page_size": 0,
            "freelist_count": 0,
            "reclaimable_bytes": 0,
            "db_bytes_delta": storage["db_bytes"]
            - _previous_value(previous, "storage", "db_bytes"),
            "recent_text_bytes_delta": -_previous_value(
                previous, "corpus", "recent_text_bytes"
            ),
        }
    )
    return {
        "version": SNAPSHOT_VERSION,
        "generated_at": generated_at,
        "state": "not_ready",
        "reason_codes": [reason],
        "corpus": {
            "window_seconds": RECENT_WINDOW_SECONDS,
            "observations": 0,
            "distinct_contexts": 0,
            "sessions": 0,
            "app_count": 0,
            "app_ids": [],
            "app_ids_truncated": False,
            "source_counts": {source: 0 for source in _SOURCES},
            "last_observation_ts": None,
            "recent_text_bytes": 0,
            "substantive_observations": 0,
            "substantive_ratio": 0.0,
            "cue_counts": {"decision": 0, "progress": 0, "open_loop": 0},
        },
        "capture": {
            "daemon": {"state": "unknown"},
            "paused": False,
            "attempts": {
                "attempts": 0,
                "finished": 0,
                "bytes_emitted": 0,
                "average_elapsed_ms": 0.0,
                "max_elapsed_ms": 0,
                "failed_bounded": 0,
                "budget_exhausted": 0,
                "reason_counts": {
                    reason: 0 for reason in sorted(CAPTURE_REASON_CODES)
                },
                "trigger_counts": {
                    trigger: 0 for trigger in sorted(CAPTURE_TRIGGERS)
                },
                "adapter_counts": {
                    adapter: 0 for adapter in sorted(CAPTURE_ADAPTERS)
                },
                "completeness_counts": {
                    value: 0 for value in sorted(CAPTURE_COMPLETENESS)
                },
                "outcome_counts": {
                    outcome: 0 for outcome in sorted(CAPTURE_OUTCOMES)
                },
            },
        },
        "storage": storage,
        "resources": {"elapsed_ms": 0.0, "peak_rss_bytes": _peak_rss_bytes()},
    }


__all__ = [
    "FRESHNESS_MAX_AGE_SECONDS",
    "RECENT_WINDOW_SECONDS",
    "SNAPSHOT_NAME",
    "evaluate_and_record_snapshot",
    "evaluate_store",
    "invalidate_snapshot",
    "read_snapshot",
    "snapshot_path",
    "unavailable_report",
    "write_snapshot",
]
