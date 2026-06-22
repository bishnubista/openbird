import SwiftUI

/// The compact connector-rail timeline (handoff Direction C): a right-aligned
/// start/end time column, a colored node on a dotted connector, and a flat text
/// entry per session — NO card (that heavier card rail belongs to Today). Session
/// formatting (app name, duration, capture count, time) routes through the shared
/// `TimelineFormatting` / `CitationFormatting` / `SourceIdentity` helpers so this rail
/// and Today's can never disagree on a rendered value.
struct SessionTimeline: View {
    let sessions: [TimelineSession]
    /// Resolve a session's app to a display name (injected from the owning model so
    /// the view never touches LaunchServices per row).
    let displayName: (String?) -> String

    @Environment(\.colorScheme) private var scheme

    private let timeColumnWidth: CGFloat = 50
    private let rowSpacing: CGFloat = 11
    private let nodeSize: CGFloat = 9
    private let nodeTop: CGFloat = 3
    /// Distance from a row's top edge to its node's center.
    private var nodeCenter: CGFloat { nodeTop + nodeSize / 2 }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(sessions.enumerated()), id: \.offset) { idx, session in
                row(session, isFirst: idx == 0, isLast: idx == sessions.count - 1)
            }
        }
    }

    @ViewBuilder
    private func row(_ session: TimelineSession, isFirst: Bool, isLast: Bool) -> some View {
        let identity = SourceIdentity.forApp(session.app)
        let name = displayName(session.app)
        let title = (session.window?.isEmpty == false) ? session.window! : name
        HStack(alignment: .top, spacing: rowSpacing) {
            // Start over end, right-aligned tabular times.
            VStack(alignment: .trailing, spacing: 0) {
                Text(CitationFormatting.shortTime(session.start))
                    .foregroundStyle(OB.textSecondary(scheme))
                Text(CitationFormatting.shortTime(session.end))
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            .font(.system(size: 11))
            .monospacedDigit()
            .frame(width: timeColumnWidth, alignment: .trailing)
            .padding(.top, 1)

            // Node circle (painted over the background connector line).
            Circle()
                .fill(identity.color)
                .frame(width: nodeSize, height: nodeSize)
                .padding(.top, nodeTop)

            // Flat content entry.
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                    .lineLimit(1)
                Text(metaLine(session, appName: name))
                    .font(.system(size: 11))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .lineLimit(1)
            }
            .padding(.bottom, isLast ? 0 : 16)

            Spacer(minLength: 0)
        }
        .fixedSize(horizontal: false, vertical: true)
        // The connector is drawn in the row BACKGROUND, which is proposed the row's
        // real rendered height — so the line fills it regardless of `.fixedSize` /
        // flexible-height propagation (Codex review: `maxHeight: .infinity` in-flow
        // does not receive a finite proposal here). Middle/first rows fill their whole
        // height so segments are contiguous; the last row draws only a short stub up to
        // its node, joining the line above it.
        .background(alignment: .topLeading) { connector(isFirst: isFirst, isLast: isLast) }
    }

    @ViewBuilder
    private func connector(isFirst: Bool, isLast: Bool) -> some View {
        let leadingInset = timeColumnWidth + rowSpacing + nodeSize / 2 - 1
        if isLast {
            if !isFirst {
                // Short fixed stub from the row top down to the node center, meeting the
                // full-height line of the row above. (A single-session row draws nothing.)
                Rectangle()
                    .fill(OB.separator(scheme))
                    .frame(width: 2, height: nodeCenter)
                    .padding(.leading, leadingInset)
            }
        } else {
            // Full row-height line (trimmed to start at the node center on the first row).
            Rectangle()
                .fill(OB.separator(scheme))
                .frame(width: 2)
                .padding(.leading, leadingInset)
                .padding(.top, isFirst ? nodeCenter : 0)
        }
    }

    private func metaLine(_ s: TimelineSession, appName: String) -> String {
        let dur = TimelineFormatting.durationLabel(s.end - s.start)
        let caps = "\(TimelineFormatting.grouped(s.count)) capture\(s.count == 1 ? "" : "s")"
        return "\(appName) · \(dur) · \(caps)"
    }
}
