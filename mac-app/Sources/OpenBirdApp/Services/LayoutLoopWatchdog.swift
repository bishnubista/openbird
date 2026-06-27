import Foundation
import os

/// Detects the SwiftUI/AppKit layout feedback loop from #169 — where the Core-Animation
/// display cycle re-lays-out the view tree every frame, the main runloop never makes
/// forward progress, and a CPU core pins at ~100% (fan, heat, battery, RSS bloat).
///
/// **It measures main-thread forward progress, not runloop phase.** A main-thread `Timer`
/// (in `.common` modes) stamps a heartbeat each tick; a background timer reports when that
/// stamp goes stale. A wedged main thread spinning in CA layout can't run its own Timer
/// block, so the heartbeat staleness is the tell — regardless of which runloop observer or
/// source the wedge lives under (the issue's stack shows it under a `beforeWaiting`
/// observer, `CA::Transaction::flush_as_runloop_observer`, which an awake/asleep classifier
/// would miss). A healthy *idle* app still fires the Timer (timers wake the runloop), so an
/// idle runloop is NOT a false positive.
///
/// **Detection only.** The wedged display cycle can't be safely broken from outside (see
/// the issue), so this logs a privacy-safe reason code (elapsed seconds, counts only — no
/// captured text, titles, or URLs) to surface the next occurrence in logs / a dev console
/// instead of via the fan. It catches main-thread *heartbeat starvation* specifically; a
/// high-CPU churn that still lets the runloop service its timers would not trip it.
///
/// Off by default in release builds; on in DEBUG, or when `OPENBIRD_LAYOUT_WATCHDOG` is set
/// to a non-empty, non-"0" value. Threshold defaults to 5s (`OPENBIRD_LAYOUT_WATCHDOG_SECONDS`).
///
/// `@unchecked Sendable`: state is partitioned by domain — `lastTick` is lock-protected
/// (the only field crossing threads), `reportedStall`/`poller` are confined to `pollQueue`,
/// and `heartbeat`/`started` are touched on the main thread. The compiler can't prove this,
/// hence `@unchecked`.
final class LayoutLoopWatchdog: @unchecked Sendable {
    private static let log = Logger(subsystem: "ai.openbird.OpenBird", category: "watchdog")

    /// Last main-thread heartbeat (monotonic `systemUptime` seconds). Written on the main
    /// thread by the heartbeat Timer, read on the background queue — hence the lock.
    private let lastTick: OSAllocatedUnfairLock<Double>

    private let tickInterval: TimeInterval
    private let pollInterval: TimeInterval
    private let threshold: TimeInterval

    private var heartbeat: Timer?
    private var poller: DispatchSourceTimer?
    private let pollQueue = DispatchQueue(label: "ai.openbird.layout-watchdog")
    /// Confined to `pollQueue` (the serial poller handler) — dedupes one stall episode so
    /// we log it once, then again only after a fresh tick clears it.
    private var reportedStall = false
    private var started = false

    init(threshold: TimeInterval = 5, tickInterval: TimeInterval = 1, pollInterval: TimeInterval = 2) {
        self.threshold = max(1, threshold)
        self.tickInterval = tickInterval
        self.pollInterval = pollInterval
        self.lastTick = OSAllocatedUnfairLock(initialState: ProcessInfo.processInfo.systemUptime)
    }

    /// Convenience init reading the threshold from the environment (clamped to >= 1s).
    convenience init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        let seconds = environment["OPENBIRD_LAYOUT_WATCHDOG_SECONDS"].flatMap(Double.init)
        self.init(threshold: seconds ?? 5)
    }

    /// Start the heartbeat + poller. Idempotent. Must be called on the main thread (it
    /// schedules the heartbeat Timer on `RunLoop.main`).
    func start() {
        dispatchPrecondition(condition: .onQueue(.main))
        guard !started else { return }
        started = true
        // Seed the heartbeat at start so the first poll can't see a stale stamp.
        lastTick.withLock { $0 = ProcessInfo.processInfo.systemUptime }

        // Capture the (Sendable) lock directly, not self: the heartbeat only stamps time.
        let lastTick = self.lastTick
        let timer = Timer(timeInterval: tickInterval, repeats: true) { _ in
            lastTick.withLock { $0 = ProcessInfo.processInfo.systemUptime }
        }
        // `.common` so the heartbeat keeps ticking during menu / live-resize tracking modes
        // (otherwise a benign mode switch would look like a stall).
        RunLoop.main.add(timer, forMode: .common)
        heartbeat = timer

        let source = DispatchSource.makeTimerSource(queue: pollQueue)
        source.schedule(deadline: .now() + pollInterval, repeating: pollInterval)
        source.setEventHandler { [weak self] in self?.poll() }
        poller = source
        source.resume()
    }

    /// Stop both timers (the app generally never needs this — the watchdog lives for the
    /// process — but it keeps the type self-contained and leak-free for tests).
    func stop() {
        heartbeat?.invalidate()
        heartbeat = nil
        poller?.cancel()
        poller = nil
        started = false
    }

    deinit { poller?.cancel(); heartbeat?.invalidate() }

    /// Runs on `pollQueue`. Reads the heartbeat, logs once per stall episode, and resets
    /// when a fresh tick clears it.
    private func poll() {
        let last = lastTick.withLock { $0 }
        let now = ProcessInfo.processInfo.systemUptime
        if let seconds = Self.stallReport(lastTick: last, now: now,
                                          threshold: threshold, alreadyReported: reportedStall) {
            reportedStall = true
            Self.log.error("layout-watchdog main-thread-stall seconds=\(seconds, privacy: .public)")
        } else if reportedStall && Self.stallCleared(lastTick: last, now: now, threshold: threshold) {
            reportedStall = false
            Self.log.info("layout-watchdog main-thread-recovered")
        }
    }

    // MARK: - Pure decision logic (unit-tested)

    /// Rounded stall seconds to log when the main thread hasn't ticked within `threshold`
    /// AND this stall episode hasn't already been reported (dedupe); `nil` otherwise.
    static func stallReport(lastTick: Double, now: Double, threshold: TimeInterval,
                            alreadyReported: Bool) -> Int? {
        let elapsed = now - lastTick
        guard elapsed >= threshold, !alreadyReported else { return nil }
        return Int(elapsed.rounded())
    }

    /// Whether a fresh heartbeat has cleared a previously-reported stall, so the next stall
    /// is allowed to log again.
    static func stallCleared(lastTick: Double, now: Double, threshold: TimeInterval) -> Bool {
        now - lastTick < threshold
    }

    /// Whether the watchdog should run. Explicit `OPENBIRD_LAYOUT_WATCHDOG` (non-empty,
    /// non-"0") wins; otherwise on in DEBUG, off in release.
    static func isEnabled(environment: [String: String], isDebug: Bool) -> Bool {
        if let value = environment["OPENBIRD_LAYOUT_WATCHDOG"] {
            return !value.isEmpty && value != "0"
        }
        return isDebug
    }
}
