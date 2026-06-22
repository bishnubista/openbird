import Foundation

/// Shared timeline number/duration formatting, so the Today card rail and the
/// compact Direction-C rail render identical labels — a value must never read
/// "2h 28m" in one rail and "148m" in another (Codex review: one behavioral
/// interpretation of sessions, not two). Time-of-day labels come from
/// `CitationFormatting.shortTime`.
enum TimelineFormatting {
    /// Group-separated integer ("1,284").
    static func grouped(_ n: Int) -> String {
        NumberFormatter.localizedString(from: NSNumber(value: n), number: .decimal)
    }

    /// Compact duration ("30s" / "5m" / "2h 28m").
    static func durationLabel(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total)s" }
        let minutes = total / 60
        if minutes < 60 { return "\(minutes)m" }
        return "\(minutes / 60)h \(minutes % 60)m"
    }
}
