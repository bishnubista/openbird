import Foundation
import os

/// Owns the Spotlight Ask panel's conversation, independently of the window
/// `ChatView`. It calls `OpenBirdService.askChat` directly (sharing only the
/// service instance) rather than observing `AppModel.chatResult`, so a result from
/// another surface can never be mis-appended to the panel's thread, and a panel ask
/// never disturbs the window chat. Read-only status (`askUnavailableReason`,
/// model status) is still sourced from the shared `AppModel` so guidance matches
/// the rest of the app.
@MainActor
final class AskPanelModel: ObservableObject {
    /// One question and its eventual grounded answer (or error).
    struct Turn: Identifiable, Equatable {
        let id = UUID()
        let question: String
        /// Day scope captured at ask time (0=today, 1=yesterday, ...), nil when
        /// unscoped. Snapshotted per turn so a later `dayScope` change cannot
        /// retarget an already in-flight ask.
        var dayScope: Int?
        var result: ChatResult?
        var error: String?
    }

    @Published private(set) var thread: [Turn] = []
    @Published private(set) var busy = false

    /// Privacy-safe outcome signpost. Emits ONLY booleans/counts/reason codes —
    /// never the question, answer, or any captured text/window title — so an ask's
    /// grounding can be asserted from `log show`/`log stream` (subsystem
    /// "ai.openbird.OpenBird", category "ask") by an automated app verifier with no
    /// human reading the screen. Mirrors the privacy rule the CLI `rag_debug` meta
    /// tier already follows.
    private static let log = Logger(subsystem: "ai.openbird.OpenBird", category: "ask")

    /// Optional day scope (0=today, 1=yesterday, ...) applied to subsequent asks in
    /// this thread. Set by the Today view's "Ask about this day" so answers are
    /// hard-scoped to the viewed day; nil for the generic/global Ask, which stays
    /// unscoped. Forwarded to the CLI as `--day N`.
    @Published var dayScope: Int?

    /// The in-flight turn. A completion whose turn is no longer active (because
    /// `clear()` ran, or a newer ask superseded it) is dropped instead of mutating
    /// `busy`/`thread`, so a slow CLI call can never pin the UI in a stale state.
    private var activeTurnID: UUID?

    /// Seam: produce a `ChatResult` for (question, dayScope), or throw. Defaults to
    /// the real CLI-backed `askChat`; tests inject a stub so no subprocess is spawned.
    private let performAsk: @Sendable (String, Int?) throws -> ChatResult
    // Strong ref: AppModel never refers back to AskPanelModel, so there is no retain
    // cycle, and a strong hold avoids a dangling `unowned` if ownership ever changes.
    private let appModel: AppModel

    init(service: OpenBirdService, appModel: AppModel) {
        self.appModel = appModel
        self.performAsk = { try service.askChat($0, dayOffset: $1) }
    }

    /// Test seam: inject the ask implementation directly. The closure receives the
    /// question and the turn's captured day scope, so tests can assert forwarding.
    init(
        appModel: AppModel,
        performAsk: @escaping @Sendable (String, Int?) throws -> ChatResult
    ) {
        self.appModel = appModel
        self.performAsk = performAsk
    }

    // Read-only display passthroughs (so the panel matches app-wide guidance).
    var askUnavailableReason: String? { appModel.askUnavailableReason }
    var localModelStatusSummary: String { appModel.localModelStatusSummary }
    var localModelStatusState: StepState { appModel.localModelStatusState }

    /// UI entry point: begin an ask and run it to completion in the background.
    /// Returns whether the ask was accepted (false on empty/whitespace, already busy,
    /// or unavailable), so a caller can keep the draft text instead of dropping it
    /// silently when submitted while busy (Codex review).
    @discardableResult
    func ask(_ question: String) -> Bool {
        guard let turn = beginAsk(question) else { return false }
        Task { await complete(turn) }
        return true
    }

    /// Validate + append the pending turn. Returns the turn to run, or nil when the
    /// ask is a no-op (empty/whitespace, already busy) or unavailable (no memory yet,
    /// in which case the guidance is recorded as that turn's error). Split out from
    /// `complete` so the control flow is unit-testable without a window or process.
    @discardableResult
    func beginAsk(_ question: String) -> Turn? {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty, !busy else { return nil }
        if let reason = appModel.askUnavailableReason {
            thread.append(Turn(question: q, result: nil, error: reason))
            return nil
        }
        busy = true
        let turn = Turn(question: q, dayScope: dayScope)   // snapshot the scope now
        activeTurnID = turn.id
        thread.append(turn)
        return turn
    }

    /// Run the (blocking) ask off the main actor and fill in *this* turn's result or
    /// error. If the turn is no longer active (cleared or superseded) when the call
    /// returns, the outcome is dropped — it must not resurrect a removed turn or
    /// clear a `busy` flag that now belongs to a newer ask.
    func complete(_ turn: Turn) async {
        let perform = performAsk
        let q = turn.question
        let scope = turn.dayScope
        let outcome: Result<ChatResult, Error>
        do {
            outcome = .success(try await Task.detached(priority: .userInitiated) {
                try perform(q, scope)
            }.value)
        } catch {
            outcome = .failure(error)
        }
        guard activeTurnID == turn.id else { return }   // cleared / superseded → drop
        switch outcome {
        case .success(let result):
            update(turn.id) { $0.result = result }
            // Counts/booleans only — never the question/answer/citation text.
            Self.log.info(
                "ask.outcome grounded=\(result.grounded ? 1 : 0, privacy: .public) citations=\(result.citations.count, privacy: .public) derived=\(result.derivedCitations.count, privacy: .public) sources=\(result.sourceCount, privacy: .public) scoped=\(scope != nil ? 1 : 0, privacy: .public)"
            )
        case .failure(let error):
            update(turn.id) { $0.error = ChatErrorPresenter.describe(error) }
            // Boolean only — the error message can carry captured CLI stderr.
            Self.log.error("ask.outcome error=1")
        }
        busy = false
        activeTurnID = nil
    }

    func clear() {
        thread.removeAll()
        activeTurnID = nil
        busy = false
    }

    private func update(_ id: UUID, _ mutate: (inout Turn) -> Void) {
        guard let i = thread.firstIndex(where: { $0.id == id }) else { return }
        mutate(&thread[i])
    }
}
