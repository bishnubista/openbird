// Scheduler — the pure capture-cadence state machine for stream mode.
//
// This type decides WHEN to capture; it never touches AX, timers, or I/O.
// The StreamEngine (executable target) feeds it trigger/tick inputs with a
// monotonic clock value and executes the actions it returns. Keeping it pure
// makes every cadence rule — debounce tiers, the >=1s floor, the force-capture
// ceiling, AFK suppression — unit-testable with a fake clock (`swift test`).
//
// Clock discipline: all comparisons use the injected MONOTONIC `now`. Wall
// time never enters this file (a wall-clock jump must not create or destroy
// capture deadlines) — the design doc's restart-boundary rule is upheld by
// construction because a new process gets a fresh Scheduler.
//
// AFK invariant (design doc + plan consensus): while AFK, NO capture fires —
// not idle ticks, not the force ceiling. Triggers received while AFK do not
// exit AFK by themselves (a background title change must never wake capture on
// an unattended machine); only HID-idle evidence supplied via `tick()` can.

/// Why a capture fired. Values are the closed vocabulary the Python daemon
/// sanitizes against (`_CAPTURE_TRIGGERS` in daemon.py) — keep in lockstep.
public enum TriggerKind: String, Encodable, Sendable {
    case appActivated = "app_activated"
    case windowChanged = "window_changed"
    case titleChanged = "title_changed"
    case focusChanged = "focus_changed"
    case typingPause = "typing_pause"
    case idleTick = "idle_tick"
    case forceCeiling = "force_ceiling"
    case returnFromAfk = "return_from_afk"
    case startup = "startup"

    /// Debounce tier for externally-observed triggers (seconds). App/window/
    /// title/focus events settle 300 ms (windows repaint, titles cascade);
    /// typing pauses settle 500 ms. Internal kinds (ticks/ceiling/startup)
    /// fire immediately and never arm a debounce.
    var debounce: Double? {
        switch self {
        case .appActivated, .windowChanged, .titleChanged, .focusChanged:
            return 0.3
        case .typingPause:
            return 0.5
        case .idleTick, .forceCeiling, .returnFromAfk, .startup:
            return nil
        }
    }
}

/// Actions the engine must perform, in order, after feeding the scheduler.
public enum SchedulerAction: Equatable, Sendable {
    /// Run one policy-gated capture now, stamped with this trigger.
    case capture(TriggerKind)
    /// Emit an `afk_transition` event (`afk` = new state).
    case afkTransition(afk: Bool, idleSeconds: Double)
    /// Emit a liveness heartbeat.
    case heartbeat
}

/// Tuning knobs, pre-clamped by the Python side (config.py) and passed via
/// argv. The scheduler trusts them but re-applies hard floors defensively.
public struct SchedulerConfig: Sendable {
    public let minGap: Double
    public let idleTick: Double
    public let forceCeiling: Double
    public let afkThreshold: Double

    public init(
        minGap: Double = 1.0,
        idleTick: Double = 5.0,
        forceCeiling: Double = 60.0,
        afkThreshold: Double = 150.0
    ) {
        // Defensive re-clamp (argv is operator-controlled but a typo'd flag
        // must not defeat the capture floor / power budget).
        self.minGap = max(1.0, minGap)
        self.idleTick = max(1.0, idleTick)
        self.forceCeiling = max(self.minGap, max(self.idleTick, forceCeiling))
        self.afkThreshold = max(30.0, afkThreshold)
    }
}

public struct Scheduler: Sendable {
    private let config: SchedulerConfig
    /// Monotonic time of the last capture the engine actually admitted
    /// (nil = none yet). A request alone never advances the cadence floor.
    private var lastCaptureAt: Double?
    /// Armed debounce: the pending trigger and its fire deadline.
    private var pending: (kind: TriggerKind, deadline: Double)?
    /// Current AFK state (starts not-AFK; the first tick corrects it).
    public private(set) var isAfk = false

    public init(config: SchedulerConfig = SchedulerConfig()) {
        self.config = config
    }

    /// Monotonic deadline for the currently armed debounce/floor, if any.
    /// StreamEngine uses this to install a one-shot wake-up instead of waiting
    /// for the slower periodic idle timer.
    public var nextDeadline: Double? { pending?.deadline }

    /// Acknowledge that the engine actually enqueued a capture walk.
    /// This is the ONLY operation that advances the minimum-gap clock.
    public mutating func captureAdmitted(at now: Double) {
        lastCaptureAt = now
    }

    /// Earliest monotonic time the next capture may run (the >=1s floor).
    private func floorTime(after now: Double) -> Double {
        guard let last = lastCaptureAt else { return now }
        return max(now, last + config.minGap)
    }

    /// Feed an externally-observed trigger (app switch, AX notification, …).
    /// Returns actions to run now; a debounced trigger returns [] and fires on
    /// a later `tick`/`pump` once its deadline passes.
    public mutating func trigger(_ kind: TriggerKind, now: Double) -> [SchedulerAction] {
        if isAfk {
            // AFK exit needs HID evidence, which arrives via tick(); an AX/
            // title trigger alone must never wake capture (unattended machine).
            return []
        }
        guard let debounce = kind.debounce else {
            return fire(kind, now: now)
        }
        let deadline = max(now + debounce, floorTime(after: now))
        // Coalesce: a newer trigger replaces the pending kind but keeps the
        // EARLIER deadline — a storm of events yields one capture, promptly.
        if let existing = pending {
            pending = (kind, min(existing.deadline, deadline))
        } else {
            pending = (kind, deadline)
        }
        return []
    }

    /// Periodic pump from the engine's idle-tick timer. `idleSeconds` is the
    /// HID-idle reading (CGEventSourceSecondsSinceLastEventType) taken this
    /// tick. Returns actions in emit order.
    public mutating func tick(now: Double, idleSeconds: Double) -> [SchedulerAction] {
        var actions: [SchedulerAction] = []

        // -- AFK state machine first: it gates every capture decision. -------
        if !isAfk && idleSeconds >= config.afkThreshold {
            isAfk = true
            pending = nil  // stale debounce must not fire on return
            actions.append(.afkTransition(afk: true, idleSeconds: idleSeconds))
            actions.append(.heartbeat)
            return actions  // no capture work while AFK, ceiling included
        }
        if isAfk {
            if idleSeconds >= config.afkThreshold {
                // Still away: heartbeat only (liveness), no AX work at all.
                actions.append(.heartbeat)
                return actions
            }
            // HID evidence of return: exit AFK, then capture (floor-gated).
            isAfk = false
            actions.append(.afkTransition(afk: false, idleSeconds: idleSeconds))
            actions.append(contentsOf: fire(.returnFromAfk, now: now))
            return actions
        }

        // -- Armed debounce takes precedence over the backstops. --------------
        // If its deadline passed, fire it; if it is still settling, do NOT let
        // the idle-tick/first-capture backstop preempt it — that would capture
        // stale context, push the >=1s floor, and starve the fresher debounced
        // trigger. The deadline is bounded (<= 0.5s debounce + floor), so this
        // wait cannot starve the ceiling.
        if let armed = pending {
            if now >= armed.deadline {
                pending = nil
                actions.append(contentsOf: fire(armed.kind, now: now))
            } else {
                actions.append(.heartbeat)
            }
            return actions
        }

        // -- Ceiling, then idle-tick backstop. --------------------------------
        let sinceLast = lastCaptureAt.map { now - $0 }
        if let gap = sinceLast, gap >= config.forceCeiling {
            actions.append(contentsOf: fire(.forceCeiling, now: now))
            return actions
        }
        if sinceLast == nil {
            // Nothing captured yet this run: the first tick captures (startup
            // already fired in the engine, but be safe under reordering).
            actions.append(contentsOf: fire(.idleTick, now: now))
            return actions
        }
        if let gap = sinceLast, gap >= config.idleTick {
            actions.append(contentsOf: fire(.idleTick, now: now))
            return actions
        }

        actions.append(.heartbeat)
        return actions
    }

    /// Order a capture now if the floor allows; otherwise arm it as pending so
    /// the next tick delivers it once the floor clears.
    private mutating func fire(_ kind: TriggerKind, now: Double) -> [SchedulerAction] {
        let earliest = floorTime(after: now)
        if earliest > now {
            // Floor not met: defer without losing the trigger.
            if let existing = pending {
                pending = (kind, min(existing.deadline, earliest))
            } else {
                pending = (kind, earliest)
            }
            return []
        }
        return [.capture(kind)]
    }
}
