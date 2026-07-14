// Content-free capture accountability protocol shared by the stream helper and
// its wire-shape tests. No field may contain captured text, titles, URLs, hashes,
// or arbitrary reason strings.

public enum CaptureAttemptStatus: String, Encodable, Sendable {
    case started
    case finished
}

public enum CaptureAttemptOutcome: String, Encodable, Sendable {
    case capturedFull = "captured_full"
    case capturedPartial = "captured_partial"
    case capturedShallow = "captured_shallow"
    case capturedUnchanged = "captured_unchanged"
    case coalescedInflight = "coalesced_inflight"
    case skippedPolicy = "skipped_policy"
    case skippedAfk = "skipped_afk"
    case skippedPaused = "skipped_paused"
    case unsupported
    case failedBounded = "failed_bounded"
}

public enum CaptureCompleteness: String, Encodable, Sendable {
    case full
    case partial
    case shallow
    case none
}

public enum CaptureAttemptReason: String, Encodable, Sendable {
    case paused
    case selfCapture = "self_capture"
    case notAllowlisted = "not_allowlisted"
    case dangerousApp = "dangerous_app"
    case privateWindow = "private_window"
    case noFrontmostApp = "no_frontmost_app"
    case noWindow = "no_window"
    case axTimeout = "ax_timeout"
    case budgetExhausted = "budget_exhausted"
    case emptyText = "empty_text"
    case unchanged
    case normalizedEmpty = "normalized_empty"
    case policyRejected = "policy_rejected"
    case ingestFailed = "ingest_failed"
}

/// One upsertable attempt event. `started` has no outcome/finished time;
/// `finished` carries a closed outcome. The Python daemon validates this again.
public struct CaptureAttemptEvent: Encodable, Sendable {
    public let type = "capture_attempt"
    public let status: CaptureAttemptStatus
    public let attemptId: String
    public let helperEpoch: String
    public let triggerSeq: UInt64
    public let triggerTs: Double
    public let startedTs: Double?
    public let finishedTs: Double?
    public let bundleId: String?
    public let trigger: TriggerKind
    public let adapterId: String?
    public let extractorVersion: String?
    public let policyTier: Int?
    public let outcome: CaptureAttemptOutcome?
    public let nodesVisited: Int
    public let bytesEmitted: Int
    public let elapsedMs: Int
    public let completeness: CaptureCompleteness?
    public let reasonCodes: [CaptureAttemptReason]
    public let coalescedTriggerCount: Int
    public let earliestCoalescedTs: Double?
    public let successorAttemptId: String?

    public init(
        status: CaptureAttemptStatus,
        attemptId: String,
        helperEpoch: String,
        triggerSeq: UInt64,
        triggerTs: Double,
        startedTs: Double? = nil,
        finishedTs: Double? = nil,
        bundleId: String? = nil,
        trigger: TriggerKind,
        adapterId: String? = nil,
        extractorVersion: String? = nil,
        policyTier: Int? = nil,
        outcome: CaptureAttemptOutcome? = nil,
        nodesVisited: Int = 0,
        bytesEmitted: Int = 0,
        elapsedMs: Int = 0,
        completeness: CaptureCompleteness? = nil,
        reasonCodes: [CaptureAttemptReason] = [],
        coalescedTriggerCount: Int = 0,
        earliestCoalescedTs: Double? = nil,
        successorAttemptId: String? = nil
    ) {
        self.status = status
        self.attemptId = attemptId
        self.helperEpoch = helperEpoch
        self.triggerSeq = triggerSeq
        self.triggerTs = triggerTs
        self.startedTs = startedTs
        self.finishedTs = finishedTs
        self.bundleId = bundleId
        self.trigger = trigger
        self.adapterId = adapterId
        self.extractorVersion = extractorVersion
        self.policyTier = policyTier
        self.outcome = outcome
        self.nodesVisited = nodesVisited
        self.bytesEmitted = bytesEmitted
        self.elapsedMs = elapsedMs
        self.completeness = completeness
        self.reasonCodes = reasonCodes
        self.coalescedTriggerCount = coalescedTriggerCount
        self.earliestCoalescedTs = earliestCoalescedTs
        self.successorAttemptId = successorAttemptId
    }

    private enum CodingKeys: String, CodingKey {
        case type, status, trigger, outcome, completeness
        case attemptId = "attempt_id"
        case helperEpoch = "helper_epoch"
        case triggerSeq = "trigger_seq"
        case triggerTs = "trigger_ts"
        case startedTs = "started_ts"
        case finishedTs = "finished_ts"
        case bundleId = "bundle_id"
        case adapterId = "adapter_id"
        case extractorVersion = "extractor_version"
        case policyTier = "policy_tier"
        case nodesVisited = "nodes_visited"
        case bytesEmitted = "bytes_emitted"
        case elapsedMs = "elapsed_ms"
        case reasonCodes = "reason_codes"
        case coalescedTriggerCount = "coalesced_trigger_count"
        case earliestCoalescedTs = "earliest_coalesced_ts"
        case successorAttemptId = "successor_attempt_id"
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        try c.encode(status, forKey: .status)
        try c.encode(attemptId, forKey: .attemptId)
        try c.encode(helperEpoch, forKey: .helperEpoch)
        try c.encode(triggerSeq, forKey: .triggerSeq)
        try c.encode(triggerTs, forKey: .triggerTs)
        try c.encodeIfPresent(startedTs, forKey: .startedTs)
        try c.encodeIfPresent(finishedTs, forKey: .finishedTs)
        try c.encodeIfPresent(bundleId, forKey: .bundleId)
        try c.encode(trigger, forKey: .trigger)
        try c.encodeIfPresent(adapterId, forKey: .adapterId)
        try c.encodeIfPresent(extractorVersion, forKey: .extractorVersion)
        try c.encodeIfPresent(policyTier, forKey: .policyTier)
        try c.encodeIfPresent(outcome, forKey: .outcome)
        try c.encode(nodesVisited, forKey: .nodesVisited)
        try c.encode(bytesEmitted, forKey: .bytesEmitted)
        try c.encode(elapsedMs, forKey: .elapsedMs)
        try c.encodeIfPresent(completeness, forKey: .completeness)
        try c.encode(reasonCodes, forKey: .reasonCodes)
        try c.encode(coalescedTriggerCount, forKey: .coalescedTriggerCount)
        try c.encodeIfPresent(earliestCoalescedTs, forKey: .earliestCoalescedTs)
        try c.encodeIfPresent(successorAttemptId, forKey: .successorAttemptId)
    }
}
