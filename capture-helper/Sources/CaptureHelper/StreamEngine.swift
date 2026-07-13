// StreamEngine — persistent event-driven capture (`--stream` mode).
//
// One long-lived process replaces the spawn-every-2s one-shot flow: NSWorkspace
// app-activation + AXObserver focus/title notifications feed the pure Scheduler
// (CaptureHelperCore), whose actions drive policy-gated captures, AFK
// transitions, and liveness heartbeats over the SAME private stdout pipe.
//
// Privacy invariants carried over unchanged from one-shot mode:
//   * captured text flows ONLY to the fstat-verified private pipe (stdout);
//   * the allowlist / dangerous-app / incognito gates run in captureFrontmost
//     BEFORE any AX text is read, on every single capture;
//   * stderr carries only reason-code diagnostics, never content;
//   * no new event type (heartbeat / afk_transition / system) carries window
//     titles, URLs, or text — metadata only.
//
// Lifecycle (fail-closed):
//   * startup: private-pipe check -> AX trust (exit 2) -> heartbeat seq=0 (the
//     stream handshake the daemon uses to detect stream support) -> observers
//     -> startup capture -> RunLoop.main.run().
//   * SIGTERM/SIGINT -> exit 0 (clean stop, daemon-initiated).
//   * stdout EPIPE (daemon died) -> exit 3 (matches the pipe-refusal code).
//   * AX trust lost mid-run (checked every tick) -> exit 2.
//
// Clock discipline: the Scheduler sees ONLY monotonic uptime (mach absolute
// time via DispatchTime); wall-clock enters events as the stored `ts` field
// only. mach time freezes during sleep — acceptable because sleep/lock emit
// `system` events and the post-wake HID idle reading immediately reflects the
// real gap (entering AFK if warranted).

import ApplicationServices
import AppKit
import CaptureHelperCore
import Foundation

// MARK: - Stream event payloads (metadata only — no content fields exist)

private struct HeartbeatEvent: Encodable {
    var type = "heartbeat"
    let ts: Double
    let seq: UInt64
    let afk: Bool
    let paused: Bool
}

private struct AfkTransitionEvent: Encodable {
    var type = "afk_transition"
    let ts: Double
    let afk: Bool
    let idle_seconds: Double
}

private struct SystemStreamEvent: Encodable {
    var type = "system"
    let ts: Double
    let kind: String
}

/// Pre-debounce app-boundary marker (Phase B spans): emitted the instant the
/// frontmost app changes, BEFORE the capture debounce settles, so span
/// boundaries are exact even when a fast A->B->A switch coalesces away B's
/// capture frame. Tier-safe by construction: bundle id only (from
/// NSRunningApplication — no AX call, no title, no URL, no text).
private struct AppChangedEvent: Encodable {
    var type = "app_changed"
    let ts: Double
    let app: String?
}

// MARK: - Locked POSIX emitter (EPIPE-aware)

/// Serializes ALL stream-mode stdout writes (walk queue vs main run loop) and
/// writes via POSIX `write(2)` so a dead reader is detectable: `FileHandle`'s
/// non-throwing write cannot report EPIPE. With SIGPIPE ignored, EPIPE surfaces
/// as errno and the helper exits 3 — the same code as the startup pipe refusal,
/// so the daemon classifies both identically.
final class StreamEmitter {
    private let lock = NSLock()
    private let encoder: JSONEncoder

    init() {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        self.encoder = enc
    }

    func emit<E: Encodable>(_ event: E) {
        // The WHOLE encode+write path is one critical section: JSONEncoder is
        // a mutable object shared between the main run loop and the walk
        // queue, so encoding outside the lock would be a data race.
        lock.lock()
        defer { lock.unlock() }
        guard var data = try? encoder.encode(event) else {
            diag("capture: encode_failed")
            return
        }
        data.append(0x0A)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            var offset = 0
            while offset < raw.count {
                let n = write(1, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                if n > 0 {
                    offset += n
                    continue
                }
                if errno == EINTR { continue }
                if errno == EPIPE {
                    diag("capture: stdout_epipe; exiting")
                    exit(3)
                }
                diag("capture: stdout_write_failed errno=\(errno)")
                exit(3)
            }
        }
    }
}

// MARK: - Engine

final class StreamEngine {
    private var scheduler: Scheduler
    private let allow: Set<String>
    private let block: Set<String>
    private let detailedCaptureApps: Set<String>
    private let pauseFile: String?
    private let captureUrls: Bool
    private let idleTick: Double
    private let emitter = StreamEmitter()

    /// Serial utility-QoS queue for AX walks: never on the main run loop, at
    /// most one walk in flight (triggers arriving mid-walk are dropped; the
    /// idle tick backstops within `idleTick` seconds).
    private let walkQueue = DispatchQueue(
        label: "openbird.capture.walk", qos: .utility)
    /// Main-thread-confined (all scheduler input arrives on the main run loop).
    private var walkInFlight = false
    private var heartbeatSeq: UInt64 = 0
    private var axObserver: AXObserver?
    private var observedPid: pid_t = -1
    private var signalSources: [DispatchSourceSignal] = []
    private var tickTimer: Timer?
    /// Mic run-state monitor (Phase C1); retained for the process lifetime.
    private var micMonitor: MicMonitor?
    /// Main-thread mirror of the aggregate mic bit (flipped in the MicMonitor
    /// callback); snapshotted into the OCR runtime at dispatch time so the
    /// walk queue never reads main-confined state.
    private var micHot = false
    /// OCR fallback opt-in (Phase C2): the per-app set from `--ocr-apps` and
    /// its min-interval. Runtime constructed once in run() when non-empty.
    private let ocrApps: Set<String>
    private let ocrMinInterval: Double
    private var ocrRuntime: OcrRuntime?

    init(
        allow: Set<String>, block: Set<String>, detailedCaptureApps: Set<String>,
        pauseFile: String?,
        captureUrls: Bool, config: SchedulerConfig,
        ocrApps: Set<String> = [], ocrMinInterval: Double = 30.0
    ) {
        self.allow = allow
        self.block = block
        self.detailedCaptureApps = detailedCaptureApps
        self.pauseFile = pauseFile
        self.captureUrls = captureUrls
        self.idleTick = config.idleTick
        self.scheduler = Scheduler(config: config)
        self.ocrApps = ocrApps
        self.ocrMinInterval = ocrMinInterval
    }

    /// Monotonic seconds (mach absolute time) — the Scheduler's only clock.
    private func mono() -> Double {
        Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000.0
    }

    /// HID seconds since the user's last input. Uses concrete event types (not
    /// the any-event pseudo-constant, which is brittle across SDKs); the MIN
    /// across them is "time since last input of any common kind". No event tap,
    /// no Input Monitoring TCC — this is a read-only counter.
    private func hidIdleSeconds() -> Double {
        let types: [CGEventType] = [
            .keyDown, .mouseMoved, .leftMouseDown, .rightMouseDown,
            .otherMouseDown, .scrollWheel,
        ]
        return types.map {
            CGEventSource.secondsSinceLastEventType(.hidSystemState, eventType: $0)
        }.min() ?? 0
    }

    // MARK: run loop

    func run(noPrompt: Bool) -> Never {
        // Fail-closed startup order (same as one-shot): private pipe, then AX
        // trust. Pause does NOT exit stream mode — heartbeats carry paused=true.
        if !stdoutIsPrivatePipe() {
            diag("capture: stdout is not a private pipe (launch via the daemon); refusing (fail-closed)")
            exit(3)
        }
        if !ensureAccessibilityTrust(prompt: !noPrompt) {
            diag("capture: accessibility_not_trusted")
            exit(2)
        }

        // Signals: SIGPIPE must be ignored so a dead daemon surfaces as EPIPE
        // (handled in StreamEmitter -> exit 3); TERM/INT are the daemon's clean
        // stop (exit 0). signal() + DispatchSourceSignal is the standard pair.
        signal(SIGPIPE, SIG_IGN)
        for sig in [SIGTERM, SIGINT] {
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            src.setEventHandler {
                diag("capture: stream_stop reason=signal")
                exit(0)
            }
            src.resume()
            signalSources.append(src)
        }

        // Bound EVERY AX round-trip process-wide at 1s (system-wide element =
        // process default). A hung target app costs at most 1s per batched
        // call; the walk then accepts partial text.
        AXUIElementSetMessagingTimeout(AXUIElementCreateSystemWide(), 1.0)

        // Stream handshake: heartbeat seq=0 BEFORE anything else. The daemon
        // uses "no heartbeat ever" to detect an old binary that ignored
        // --stream (auto-downgrade to one-shot polling), and this satisfies
        // the first-data-<5s budget.
        emitHeartbeat()

        // Mic run-state monitor (Phase C1): aggregate input-device run-state
        // flips become `system` events (mic_started / mic_stopped) — the raw
        // hardware signal only; the meeting JUDGMENT is daemon-side policy.
        // Metadata bit, no audio read, no TCC. Stream-mode only by nature
        // (the one-shot helper has no process to host listeners).
        let mic = MicMonitor(
            hal: CoreAudioMicHAL(),
            onFlip: { [weak self] hot in
                // Main-thread by MicMonitor contract: mirror the bit for the
                // OCR gate's mic-hot suppression, then emit the edge event.
                self?.micHot = hot
                // Live-propagate into the OCR runtime (lock-backed setter):
                // the OCR decision must see mic flips that happen mid-walk.
                self?.ocrRuntime?.micHot = hot
                self?.emitSystem(hot ? "mic_started" : "mic_stopped")
            },
            diag: { diag($0) })
        mic.start()
        micMonitor = mic

        // OCR fallback runtime (Phase C2): constructed once, only when the
        // user opted apps in. The first tccGranted() reading emits the startup
        // `ocr_available` / `ocr_unavailable` system event (metadata only);
        // later flips re-emit at OCR-attempt time. Preflight only — the
        // helper NEVER triggers the Screen Recording prompt (the grant flow
        // lives in the mac-app).
        if !ocrApps.isEmpty {
            let runtime = OcrRuntime(
                apps: ocrApps,
                minInterval: ocrMinInterval,
                hal: makeOcrHAL(),
                tccPreflight: { CGPreflightScreenCaptureAccess() },
                emitSystem: { [weak self] kind in self?.emitSystem(kind) })
            _ = runtime.tccGranted()
            ocrRuntime = runtime
        }

        installWorkspaceObservers()
        if let front = NSWorkspace.shared.frontmostApplication {
            rebuildAXObserver(pid: front.processIdentifier)
        }

        // First content capture, immediately (floor-exempt: nothing captured yet).
        perform(scheduler.trigger(.startup, now: mono()))

        // Idle tick: the poll backstop + AFK detector + liveness pump. Tolerant
        // timer so the OS can coalesce wakeups (battery).
        let timer = Timer(timeInterval: idleTick, repeats: true) { [weak self] _ in
            self?.onTick()
        }
        timer.tolerance = idleTick * 0.2
        RunLoop.main.add(timer, forMode: .common)
        tickTimer = timer

        RunLoop.main.run()  // blocks; exit paths are the signal/EPIPE/trust handlers
        exit(0)
    }

    // MARK: inputs (all on the main run loop)

    private func onTick() {
        // TCC fail-closed: a revoked grant must stop capture promptly, not
        // silently yield empty walks until the next restart.
        if !AXIsProcessTrusted() {
            diag("capture: accessibility_trust_lost")
            exit(2)
        }
        perform(scheduler.tick(now: mono(), idleSeconds: hidIdleSeconds()))
    }

    private func onTrigger(_ kind: TriggerKind) {
        perform(scheduler.trigger(kind, now: mono()))
    }

    private func installWorkspaceObservers() {
        let wsCenter = NSWorkspace.shared.notificationCenter
        wsCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil, queue: .main
        ) { [weak self] note in
            guard let self else { return }
            if let app = note.userInfo?[NSWorkspace.applicationUserInfoKey]
                as? NSRunningApplication {
                // Span boundary marker FIRST (pre-debounce, exact timing),
                // then the debounced content trigger.
                self.emitter.emit(
                    AppChangedEvent(
                        ts: Date().timeIntervalSince1970,
                        app: app.bundleIdentifier))
                self.rebuildAXObserver(pid: app.processIdentifier)
            }
            self.onTrigger(.appActivated)
        }
        wsCenter.addObserver(
            forName: NSWorkspace.willSleepNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.emitSystem("will_sleep") }
        wsCenter.addObserver(
            forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.emitSystem("did_wake") }

        // Screen lock/unlock are distributed notifications (no NSWorkspace API).
        let dnc = DistributedNotificationCenter.default()
        dnc.addObserver(
            forName: Notification.Name("com.apple.screenIsLocked"),
            object: nil, queue: .main
        ) { [weak self] _ in self?.emitSystem("screen_locked") }
        dnc.addObserver(
            forName: Notification.Name("com.apple.screenIsUnlocked"),
            object: nil, queue: .main
        ) { [weak self] _ in self?.emitSystem("screen_unlocked") }
    }

    /// (Re)attach the AXObserver to the newly-activated app. Failure is
    /// non-fatal by design: Chromium/Electron under-emit or reject observers,
    /// and the idle tick + force ceiling backstop them.
    private func rebuildAXObserver(pid: pid_t) {
        if pid == observedPid, axObserver != nil { return }
        if let old = axObserver {
            CFRunLoopRemoveSource(
                CFRunLoopGetMain(), AXObserverGetRunLoopSource(old), .defaultMode)
        }
        axObserver = nil
        observedPid = -1

        var created: AXObserver?
        guard AXObserverCreate(pid, streamAXCallback, &created) == .success,
              let observer = created
        else {
            diag("capture: ax_observer_unavailable")
            return
        }
        let appElement = AXUIElementCreateApplication(pid)
        let refcon = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        for note in [
            kAXFocusedWindowChangedNotification,
            kAXTitleChangedNotification,
            kAXFocusedUIElementChangedNotification,
        ] {
            // Per-notification failures are fine (some apps expose only a subset).
            AXObserverAddNotification(observer, appElement, note as CFString, refcon)
        }
        CFRunLoopAddSource(
            CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .defaultMode)
        axObserver = observer
        observedPid = pid
    }

    fileprivate func handleAXNotification(_ name: String) {
        switch name {
        case String(kAXFocusedWindowChangedNotification):
            onTrigger(.windowChanged)
        case String(kAXTitleChangedNotification):
            onTrigger(.titleChanged)
        case String(kAXFocusedUIElementChangedNotification):
            onTrigger(.focusChanged)
        default:
            break
        }
    }

    // MARK: actions

    private func perform(_ actions: [SchedulerAction]) {
        for action in actions {
            switch action {
            case .capture(let kind):
                dispatchCapture(kind)
            case .afkTransition(let afk, let idleSeconds):
                emitter.emit(
                    AfkTransitionEvent(
                        // Backdate the away-boundary to when input actually
                        // stopped; the return boundary is "now".
                        ts: Date().timeIntervalSince1970 - (afk ? idleSeconds : 0),
                        afk: afk,
                        idle_seconds: idleSeconds))
            case .heartbeat:
                emitHeartbeat()
            }
        }
    }

    private func dispatchCapture(_ kind: TriggerKind) {
        // While paused, skip the walk entirely (heartbeats carry paused=true);
        // captureFrontmost re-checks the pause file anyway (defense-in-depth).
        if capturePaused(pauseFile) { return }
        if walkInFlight { return }  // next tick backstops the dropped trigger
        walkInFlight = true
        let trigger = kind.rawValue
        walkQueue.async { [weak self] in
            guard let self else { return }
            let activity = ProcessInfo.processInfo.beginActivity(
                options: .background, reason: "capture")
            captureFrontmost(
                allow: self.allow, block: self.block,
                detailedCaptureApps: self.detailedCaptureApps,
                pauseFile: self.pauseFile,
                captureUrls: self.captureUrls, trigger: trigger,
                emitter: { self.emitter.emit($0) },
                ocr: self.ocrRuntime)
            ProcessInfo.processInfo.endActivity(activity)
            DispatchQueue.main.async { self.walkInFlight = false }
        }
    }

    private func emitHeartbeat() {
        emitter.emit(
            HeartbeatEvent(
                ts: Date().timeIntervalSince1970,
                seq: heartbeatSeq,
                afk: scheduler.isAfk,
                paused: capturePaused(pauseFile)))
        heartbeatSeq += 1
    }

    private func emitSystem(_ kind: String) {
        // Reason-code metadata only; Phase B will consume these for span
        // force-close. Also worth a heartbeat's liveness credit on the daemon
        // side (it counts system events as liveness).
        emitter.emit(
            SystemStreamEvent(ts: Date().timeIntervalSince1970, kind: kind))
    }
}

/// C-function AX callback: bounce into the engine via refcon. Runs on the main
/// run loop (the observer's source is scheduled there).
private let streamAXCallback: AXObserverCallback = { _, _, notification, refcon in
    guard let refcon else { return }
    let engine = Unmanaged<StreamEngine>.fromOpaque(refcon).takeUnretainedValue()
    engine.handleAXNotification(notification as String)
}
