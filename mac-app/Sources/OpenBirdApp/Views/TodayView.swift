import SwiftUI

/// The Today/day view (handoff §3): a day's grounded briefing card, stat chips, and
/// a session timeline on a connector rail — rendered over the `timeline`/`briefing`
/// CLI JSON via `TodayModel`.
struct TodayView: View {
    @ObservedObject var model: TodayModel
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: OB.Space.l) {
                header
                briefingCard
                if let timeline = model.timeline {
                    statChips(timeline)
                }
                timelineSection
            }
            .padding(OB.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 540, minHeight: 480)
        .background(GlassBackdrop())
        .task {
            if model.timeline == nil {
                await model.load()
            }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.dayTitle)
                    .font(.system(size: 21, weight: .bold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(model.daySubtitle)
                    .font(.subheadline)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            Spacer()
            Button {
                Task { await model.refresh() }
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .disabled(model.loadingTimeline)
        }
    }

    // MARK: Daily briefing card

    private var briefingCard: some View {
        VStack(alignment: .leading, spacing: OB.Space.sm) {
            Text("DAILY BRIEFING")
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.5)
                .foregroundStyle(OB.accent)
            if model.loadingBriefing {
                HStack(spacing: OB.Space.sm) {
                    ProgressView().controlSize(.small)
                    Text("Generating briefing…")
                        .font(.callout)
                        .foregroundStyle(OB.textSecondary(scheme))
                }
            } else if let briefing = model.briefing, !briefing.isEmpty {
                Text(briefing)
                    .font(.system(size: 14))
                    .foregroundStyle(OB.textPrimary(scheme))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("No briefing available for this day.")
                    .font(.callout)
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    // MARK: Stat chips

    private func statChips(_ timeline: DayTimeline) -> some View {
        HStack(spacing: OB.Space.sm) {
            statChip("\(timeline.distinctApps)", "apps")
            statChip("\(timeline.totalObservations)", "captures")
            statChip(Self.durationLabel(timeline.activeSeconds), "active")
        }
    }

    private func statChip(_ value: String, _ label: String) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(OB.textPrimary(scheme))
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(OB.textSecondary(scheme))
        }
        .padding(.horizontal, OB.Space.m)
        .padding(.vertical, OB.Space.sm)
        .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.control))
    }

    // MARK: Session timeline

    @ViewBuilder
    private var timelineSection: some View {
        if model.loadingTimeline && model.timeline == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading timeline…").foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let error = model.timelineError {
            Label(error, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        } else if let timeline = model.timeline, !timeline.sessions.isEmpty {
            // Index-keyed: the backend buckets legacy NULL-session rows by observation
            // id, but the JSON drops that key, so element identity isn't guaranteed
            // unique. The list is render-only (no reordering), so the index is a safe,
            // stable id and avoids same-app/same-ts collisions.
            VStack(alignment: .leading, spacing: OB.Space.sm) {
                ForEach(Array(timeline.sessions.enumerated()), id: \.offset) { _, session in
                    sessionRow(session)
                }
            }
            // Continuous connector rail behind the nodes. Drawn as a background of
            // the (finite-height) sessions stack — reliable, unlike a per-row
            // Rectangle(maxHeight: .infinity) which has no bounded row to fill.
            .background(alignment: .topLeading) {
                Rectangle()
                    .fill(OB.separator(scheme))
                    .frame(width: 1.5)
                    .padding(.leading, 4.25)   // align under the 10pt node (center ≈ 5)
                    .padding(.vertical, 18)
            }
        } else {
            Text("No capture sessions for this day.")
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    private func sessionRow(_ session: TimelineSession) -> some View {
        let identity = SourceIdentity.forApp(session.app)
        return HStack(alignment: .center, spacing: OB.Space.m) {
            // Node on the connector rail; the rail line is drawn behind the stack.
            Circle().fill(identity.color).frame(width: 10, height: 10)

            HStack(spacing: OB.Space.m) {
                Text(identity.glyph)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 30, height: 30)
                    .background(identity.color, in: RoundedRectangle(cornerRadius: OB.Radius.control))
                VStack(alignment: .leading, spacing: 2) {
                    Text(Self.appName(session.app))
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    Text(Self.sessionDetail(session))
                        .font(.system(size: 11.5))
                        .foregroundStyle(OB.textSecondary(scheme))
                }
                Spacer()
                Text("\(session.count)")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            .padding(OB.Space.m)
            .glassSurface(cornerRadius: OB.Radius.card)
        }
    }

    // MARK: Formatting

    static func appName(_ bundleID: String?) -> String {
        guard let bundleID, !bundleID.isEmpty else { return "Unknown" }
        return bundleID.split(separator: ".").last.map(String.init) ?? bundleID
    }

    static func sessionDetail(_ session: TimelineSession) -> String {
        let span = "\(CitationFormatting.shortTime(session.start)) – \(CitationFormatting.shortTime(session.end))"
        return "\(span) · \(durationLabel(session.end - session.start))"
    }

    static func durationLabel(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        if total < 60 { return "\(total)s" }
        let minutes = total / 60
        if minutes < 60 { return "\(minutes)m" }
        return "\(minutes / 60)h \(minutes % 60)m"
    }
}
