import SwiftUI

/// The Today/day view (handoff §3): a 222px nav sidebar plus a main column with a
/// grounded briefing card, stat chips, and a session timeline on a connector rail —
/// rendered over the `timeline`/`briefing` CLI JSON via `TodayModel`. Capture/
/// allowlist state for the sidebar footer comes from the shared `AppModel`.
struct TodayView: View {
    @ObservedObject var model: TodayModel
    @ObservedObject var appModel: AppModel
    /// Summon the Spotlight Ask panel (the header "Ask about this day" button).
    var onAsk: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 0) {
            TodaySidebar(appModel: appModel, onAsk: onAsk)
            mainColumn
        }
        .frame(minWidth: 860, minHeight: 560)
        .background(GlassBackdrop())
        .background(WindowConfigurator())   // draggable by background under the hidden titlebar
        .task {
            if model.timeline == nil {
                await model.load()
            }
        }
    }

    private var mainColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: OB.Space.l) {
                header
                briefingCard
                timelineSection
            }
            .padding(OB.Space.xl)
            .frame(maxWidth: .infinity, alignment: .leading)
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
            askButton
        }
    }

    private var askButton: some View {
        Button(action: onAsk) {
            HStack(spacing: OB.Space.s) {
                BirdLogo().fill(OB.accent).frame(width: 15, height: 15)
                Text("Ask about this day")
                    .font(.system(size: 13, weight: .medium))
            }
            .foregroundStyle(OB.textPrimary(scheme))
            .padding(.horizontal, OB.Space.ml)
            .padding(.vertical, OB.Space.sm)
            .background(OB.fieldFill(scheme),
                        in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous)
                    .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: Daily briefing card

    private var briefingCard: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            HStack(alignment: .firstTextBaseline) {
                Text("DAILY BRIEFING")
                    .font(.system(size: 10.5, weight: .bold))
                    .tracking(0.6)
                    .foregroundStyle(OB.accent)
                Spacer()
                if let at = model.briefingGeneratedAt {
                    Text("generated \(at.formatted(date: .omitted, time: .shortened))")
                        .font(.system(size: 12))
                        .monospacedDigit()
                        .foregroundStyle(OB.textTertiary(scheme))
                }
            }
            briefingBody
            if let timeline = model.timeline {
                statChips(timeline)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(OB.Space.l)
        .glassSurface(cornerRadius: OB.Radius.card)
    }

    @ViewBuilder
    private var briefingBody: some View {
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
                .lineSpacing(4)
                .foregroundStyle(OB.textPrimary(scheme))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Text("No briefing available for this day.")
                .font(.callout)
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    // MARK: Stat chips (pill, bordered, tabular)

    private func statChips(_ timeline: DayTimeline) -> some View {
        // Three HONEST chips: apps, captures, active time. The handoff's "1 meeting"
        // chip is intentionally omitted — there is no real meeting-evidence source
        // yet (meetings is a stub), so claiming a meeting count would be fabricated.
        HStack(spacing: OB.Space.sm) {
            statChip("\(timeline.distinctApps) app\(timeline.distinctApps == 1 ? "" : "s")")
            statChip("\(Self.grouped(timeline.totalObservations)) captures")
            statChip("\(Self.durationLabel(timeline.activeSeconds)) active")
        }
    }

    private func statChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 12))
            .monospacedDigit()
            .foregroundStyle(OB.textSecondary(scheme))
            .padding(.horizontal, OB.Space.m)
            .padding(.vertical, OB.Space.s)
            .background(OB.fieldFill(scheme), in: Capsule())
            .overlay(Capsule().strokeBorder(OB.separator(scheme), lineWidth: 0.5))
    }

    // MARK: Session timeline

    @ViewBuilder
    private var timelineSection: some View {
        VStack(alignment: .leading, spacing: OB.Space.m) {
            Text("TIMELINE")
                .font(.system(size: 10.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(OB.textSecondary(scheme))
            timelineBody
        }
    }

    @ViewBuilder
    private var timelineBody: some View {
        if model.loadingTimeline && model.timeline == nil {
            HStack(spacing: OB.Space.sm) {
                ProgressView().controlSize(.small)
                Text("Loading timeline…").foregroundStyle(OB.textSecondary(scheme))
            }
        } else if let error = model.timelineError {
            Label(error, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        } else if let timeline = model.timeline, !timeline.sessions.isEmpty {
            sessionList(timeline)
        } else {
            Text("No capture sessions for this day.")
                .foregroundStyle(OB.textSecondary(scheme))
        }
    }

    private func sessionList(_ timeline: DayTimeline) -> some View {
        // Index-keyed: NULL-session rows can share app/ts, so element identity isn't
        // guaranteed unique; the list is render-only, so the index is a safe id.
        VStack(alignment: .leading, spacing: OB.Space.sm) {
            ForEach(Array(timeline.sessions.enumerated()), id: \.offset) { _, session in
                sessionRow(session)
            }
        }
        // Continuous connector rail behind the nodes. Positioned under the node
        // column (time column width + spacing + node center).
        .background(alignment: .topLeading) {
            Rectangle()
                .fill(OB.separator(scheme))
                .frame(width: 1.5)
                .padding(.leading, Self.timeColumnWidth + OB.Space.m + 5 - 0.75)
                .padding(.vertical, 20)
        }
    }

    private func sessionRow(_ session: TimelineSession) -> some View {
        let identity = SourceIdentity.forApp(session.app)
        let appName = model.displayName(session.app)
        let title = (session.window?.isEmpty == false) ? session.window! : appName
        return HStack(alignment: .top, spacing: OB.Space.m) {
            // Left time column: start over end.
            VStack(alignment: .trailing, spacing: 0) {
                Text(CitationFormatting.shortTime(session.start))
                    .foregroundStyle(OB.textSecondary(scheme))
                Text(CitationFormatting.shortTime(session.end))
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            .font(.system(size: 12))
            .monospacedDigit()
            .lineLimit(1)                       // never wrap the " AM"/" PM" onto a second line
            .frame(width: Self.timeColumnWidth, alignment: .trailing)
            .padding(.top, 14)

            // Node on the connector rail.
            Circle().fill(identity.color).frame(width: 10, height: 10).padding(.top, 18)

            // Session card.
            HStack(alignment: .top, spacing: OB.Space.m) {
                Text(identity.glyph)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(identity.color,
                                in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                        .lineLimit(1)
                    Text("\(appName) · \(Self.durationLabel(session.end - session.start))")
                        .font(.system(size: 12))
                        .foregroundStyle(OB.textSecondary(scheme))
                }
                Spacer()
                Text("\(Self.grouped(session.count)) captures")
                    .font(.system(size: 12))
                    .monospacedDigit()
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            .padding(OB.Space.m)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassSurface(cornerRadius: OB.Radius.card)
        }
    }

    // MARK: Formatting

    // Wide enough for a two-digit 12-hour time with day period ("10:52 PM") at 12pt
    // monospaced-digit — the gutter must not wrap (the connector rail inset derives
    // from this constant, so it follows automatically).
    static let timeColumnWidth: CGFloat = 62

    /// Group-separated integer ("1,284"). Forwards to the shared formatter so the
    /// Today card rail and the compact Direction-C rail never disagree on a value.
    static func grouped(_ n: Int) -> String { TimelineFormatting.grouped(n) }

    static func durationLabel(_ seconds: Double) -> String { TimelineFormatting.durationLabel(seconds) }
}
