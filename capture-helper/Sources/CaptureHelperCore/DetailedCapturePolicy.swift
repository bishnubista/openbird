import Foundation

/// Exact per-app grant used to override terminal/editor blocking at the source.
/// Pattern-looking entries are inert so a broad grant can never widen capture.
public func hasExactDetailedCaptureGrant(
    bundleId: String?, entries: Set<String>
) -> Bool {
    guard let bundleId else { return false }
    return entries.contains { entry in
        guard !entry.hasPrefix("glob:"), !entry.hasPrefix("re:") else { return false }
        return entry.caseInsensitiveCompare(bundleId) == .orderedSame
    }
}
