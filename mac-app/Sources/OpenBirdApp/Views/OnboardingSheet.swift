import SwiftUI

/// First-run onboarding sheet (handoff §4 / screenshot 06): a centered 540px glass
/// sheet with the OpenBird mark, a privacy-forward subhead, four permission rows,
/// and a primary "Start capturing" action. This is a SEPARATE first-run surface —
/// the functional, ongoing-config `SetupView` still lives in the main window.
///
/// Every row reflects REAL `AppModel` permission/model state; nothing here is
/// decorative. Rows 1 and 2 both surface the macOS Accessibility grant because in
/// OpenBird screen-text capture IS performed via Accessibility — one grant,
/// described from two angles, exactly as the design intends.
struct OnboardingSheet: View {
    @ObservedObject var model: AppModel
    @Binding var isPresented: Bool
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(spacing: OB.Space.l) {
            BirdLogo().fill(OB.accent).frame(width: 54, height: 54)

            VStack(spacing: OB.Space.sm) {
                Text("Welcome to OpenBird")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text("Local-first memory for your Mac. It reads the **text** of your active window — never screenshots. \(model.privacyStorageSummary) \(model.privacyTransmissionSummary)")
                    .font(.system(size: 13.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 420)
            }

            rows

            Button(action: start) {
                Text("Start capturing")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, OB.Space.m)
                    .background(OB.accent, in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
            }
            .buttonStyle(.plain)

            HStack(spacing: OB.Space.s) {
                Image(systemName: "lock.fill").font(.system(size: 10))
                Text("\(model.privacyTransmissionSummary) Pause anytime from the menu bar.")
                    .font(.system(size: 11.5))
            }
            .foregroundStyle(OB.textTertiary(scheme))
        }
        .padding(OB.Space.xl)
        .frame(width: 540)
        .background(GlassBackdrop())
        .task { await model.refresh() }
    }

    // MARK: Rows

    private var rows: some View {
        VStack(spacing: OB.Space.sm) {
            OnboardingRow(
                icon: "text.viewfinder",
                title: "Screen text capture",
                subtitle: "Reads active-window text via Accessibility",
                state: model.accessibilityState,
                enableTitle: "Enable",
                onEnable: { model.requestAccessibility() }
            )
            OnboardingRow(
                icon: "accessibility",
                title: "Accessibility",
                subtitle: "App & window metadata for the timeline",
                state: model.accessibilityState,
                enableTitle: "Enable",
                onEnable: { model.requestAccessibility() }
            )
            OnboardingRow(
                icon: "mic",
                title: "System audio for meetings",
                subtitle: "Transcribe calls locally · optional",
                state: model.screenRecordingState,
                enableTitle: "Enable",
                onEnable: { model.requestScreenRecording() }
            )
            OnboardingRow(
                icon: "cpu",
                title: "Active model route",
                subtitle: model.localModelStatusSummary,
                state: model.localModelStatusState,
                connectedLabel: "Connected",
                enableTitle: modelRouteEnableTitle,
                onEnable: modelRouteEnable
            )
        }
    }

    private var modelRouteEnableTitle: String? {
        if model.hasRemoteModelRoute { return nil }
        if model.report.ollamaReachable == false { return "Get Ollama" }
        if !model.report.missingModels.isEmpty { return "Pull" }
        return nil
    }

    private func modelRouteEnable() {
        if model.report.ollamaReachable == false {
            NSWorkspace.shared.open(URL(string: "https://ollama.com")!)
        } else if !model.report.missingModels.isEmpty {
            Task { await model.pullMissingModels() }
        }
    }

    private func start() {
        // Only start the capture daemon when there's something to capture — an empty
        // allowlist would launch a no-op daemon (capture is allowlist-first). Either
        // way we dismiss to the main window, which carries the full SetupView +
        // allowlist editor, so the setup path is never hidden behind a one-time sheet.
        if !model.allowlist.isEmpty {
            model.startCapture()
        }
        isPresented = false
    }
}

/// One onboarding permission row: an accent-tinted icon tile, a title + subtitle,
/// and a trailing status that is EITHER a green "Granted"/"Connected" affirmation
/// or an accent action button — driven by the real `StepState`.
private struct OnboardingRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let state: StepState
    /// When set and `state == .ok`, render a green-dot "● <label>" instead of the
    /// "✓ Granted" checkmark (used for the model-connection row).
    var connectedLabel: String? = nil
    let enableTitle: String?
    let onEnable: () -> Void

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: OB.Space.m) {
            ZStack {
                RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous)
                    .fill(OB.accent.opacity(0.14))
                Image(systemName: icon)
                    .font(.system(size: 15))
                    .foregroundStyle(OB.accent)
            }
            .frame(width: 36, height: 36)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(subtitle)
                    .font(.system(size: 12))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .lineLimit(1)
            }
            Spacer()
            status
        }
        .padding(OB.Space.m)
        .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: OB.Radius.card, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: OB.Radius.card, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private var status: some View {
        if state == .ok {
            if let connectedLabel {
                HStack(spacing: OB.Space.s) {
                    Circle().fill(OB.ok(scheme)).frame(width: 7, height: 7)
                    Text(connectedLabel)
                        .font(.system(size: 13))
                        .foregroundStyle(OB.ok(scheme))
                }
            } else {
                Label("Granted", systemImage: "checkmark")
                    .font(.system(size: 13))
                    .foregroundStyle(OB.ok(scheme))
            }
        } else {
            if let enableTitle {
                Button(action: onEnable) {
                    Text(enableTitle)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, OB.Space.m)
                        .padding(.vertical, OB.Space.s)
                        .background(OB.accent, in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
                }
                .buttonStyle(.plain)
            } else {
                Label("Review", systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 13))
                    .foregroundStyle(.orange)
            }
        }
    }
}
