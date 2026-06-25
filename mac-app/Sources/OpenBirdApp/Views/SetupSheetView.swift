import AppKit
import SwiftUI

/// The pixel-faithful OpenBird setup surface (handoff §4 / screenshot
/// `06-setup-onboarding-dark.png`): a 540pt Liquid-Glass sheet with a traffic-light
/// header, the OpenBird mark, a privacy-forward subhead, four permission rows, a primary
/// action, and a truthful privacy footer.
///
/// Rendered by the first-run `OnboardingSheet` as the install-moment modal. (The durable
/// configuration surface is the separate `SettingsView` pane.)
///
/// Every row reflects REAL `AppModel` state; nothing here is decorative except the
/// traffic-light motif. Rows 1 and 2 both surface the macOS Accessibility grant because
/// OpenBird's screen-text capture IS performed via Accessibility — one grant described
/// from two angles, exactly as the design intends.
struct SetupSheetView: View {
    @ObservedObject var model: AppModel
    /// Primary-button action. Onboarding starts capture + dismisses; the Setup tab just
    /// starts capture. The label stays "Start capturing" in both.
    var onPrimary: () -> Void
    /// The design's window-chrome dots. Purely decorative (NOT live window controls) — a
    /// "this is a glass window" motif that is part of the pixel spec. The host window's
    /// real traffic lights sit in its top-left corner, away from this centered sheet.
    var showTrafficLights: Bool = true

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        VStack(spacing: 0) {
            if showTrafficLights { trafficLights }
            content
        }
        .frame(width: 540)
        .glassSurface(cornerRadius: OB.Radius.setup)   // 14
    }

    // MARK: Traffic-light strip (38pt tall, 14pt sides, three 12pt dots, 8pt gap, leading)

    private var trafficLights: some View {
        HStack(spacing: OB.Space.sm) {                 // 8
            trafficDot(Color(hex: 0xFF5F57))
            trafficDot(Color(hex: 0xFEBC2E))
            trafficDot(Color(hex: 0x28C840))
            Spacer(minLength: 0)
        }
        .padding(.horizontal, OB.Space.ml)             // 14
        .frame(height: 38)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func trafficDot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 12, height: 12)
    }

    // MARK: Sheet body (14 top / 38 sides / 32 bottom; centered column)

    private var content: some View {
        VStack(spacing: 0) {
            BirdLogo().fill(OB.accent).frame(width: 54, height: 54)

            Text("Welcome to OpenBird")
                .font(.system(size: 23, weight: .bold))
                .foregroundStyle(OB.textPrimary(scheme))
                .padding(.top, OB.Space.ml)            // 14
                .padding(.bottom, 5)

            // The "reads text, never screenshots" promise is fixed; the storage and
            // transmission summaries are appended LIVE so this copy can never claim
            // local-only behavior for a remote route (project privacy-truthfulness rule).
            Text("Local-first memory for your Mac. It reads the **text** of your active window — never screenshots. \(model.privacyStorageSummary) \(model.privacyTransmissionSummary)")
                .font(.system(size: 13.5))
                .foregroundStyle(OB.textSecondary(scheme))
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 400)
                .padding(.bottom, OB.Space.xl)         // 22

            rows

            Button(action: onPrimary) {
                Text("Start capturing")
                    .font(.system(size: 14.5, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .background(OB.accent, in: RoundedRectangle(cornerRadius: OB.Radius.card, style: .continuous))   // 11
            }
            .buttonStyle(.plain)
            .padding(.top, OB.Space.xl)                // 22

            HStack(spacing: OB.Space.s) {              // 6
                Image(systemName: "lock.fill").font(.system(size: 11))
                Text("\(model.privacyTransmissionSummary) Pause anytime from the menu bar.")
                    .font(.system(size: 11.5))
            }
            .foregroundStyle(OB.textTertiary(scheme))
            .padding(.top, 13)
        }
        .padding(.top, OB.Space.ml)                    // 14
        .padding(.horizontal, 38)
        .padding(.bottom, 32)
        .frame(maxWidth: .infinity)
    }

    // MARK: Permission rows (10pt gap)

    private var rows: some View {
        VStack(spacing: 10) {
            SetupPermissionRow(
                icon: "text.viewfinder",
                title: "Screen text capture",
                subtitle: "Reads active-window text via Accessibility",
                state: model.accessibilityState,
                enableTitle: "Enable",
                onEnable: { model.requestAccessibility() }
            )
            SetupPermissionRow(
                icon: "accessibility",
                title: "Accessibility",
                subtitle: "App & window metadata for the timeline",
                state: model.accessibilityState,
                enableTitle: "Enable",
                onEnable: { model.requestAccessibility() }
            )
            SetupPermissionRow(
                icon: "mic",
                title: "System audio for meetings",
                subtitle: "Capture shared audio for local transcription · optional",
                state: model.screenRecordingState,
                enableTitle: "Enable",
                onEnable: { model.requestScreenRecording() }
            )
            SetupPermissionRow(
                icon: "waveform",
                title: "Meeting transcription",
                subtitle: model.meetingTranscriptionSummary,
                state: model.meetingTranscriptionState,
                connectedLabel: "Ready",
                enableTitle: nil,
                onEnable: {}
            )
            SetupPermissionRow(
                icon: "cpu",
                title: "Active model route",
                subtitle: model.localModelStatusSummary,
                state: model.localModelStatusState,
                connectedLabel: "Connected",
                enableTitle: model.modelRouteActionLabel,
                onEnable: { model.performModelRouteAction { NSWorkspace.shared.open($0) } }
            )
        }
    }
}

/// One setup permission row (handoff §4): an accent-tinted 36pt icon tile, a title +
/// subtitle, and a trailing status that is EITHER a green "Granted"/"Connected"
/// affirmation or an accent action button — driven entirely by the real `StepState`.
struct SetupPermissionRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let state: StepState
    /// When set and `state == .ok`, render a green-dot "● <label>" instead of the
    /// "✓ Granted" checkmark (used for the model-connection row).
    var connectedLabel: String? = nil
    /// Accent action shown while the step is unsatisfied. `nil` → a neutral "Review".
    let enableTitle: String?
    let onEnable: () -> Void

    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: 13) {
            ZStack {
                RoundedRectangle(cornerRadius: OB.Radius.tile, style: .continuous)   // 9
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
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    // The model-route subtitle is live status text that can be long —
                    // let it wrap instead of truncating.
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: OB.Space.sm)
            status
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 13)
        .background(
            OB.fieldFill(scheme),
            in: RoundedRectangle(cornerRadius: OB.Radius.permissionRow, style: .continuous)   // 12
        )
        .overlay(
            RoundedRectangle(cornerRadius: OB.Radius.permissionRow, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private var status: some View {
        if state == .ok {
            if let connectedLabel {
                HStack(spacing: 5) {
                    Circle().fill(OB.ok(scheme)).frame(width: 7, height: 7)
                    Text(connectedLabel)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(OB.ok(scheme))
                }
            } else {
                HStack(spacing: 5) {
                    Image(systemName: "checkmark").font(.system(size: 12.5, weight: .bold))
                    Text("Granted").font(.system(size: 12.5, weight: .semibold))
                }
                .foregroundStyle(OB.ok(scheme))
            }
        } else if let enableTitle {
            Button(action: onEnable) {
                Text(enableTitle)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, OB.Space.ml)    // 14
                    .padding(.vertical, 7)
                    .background(OB.accent, in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))   // 8
            }
            .buttonStyle(.plain)
        } else {
            HStack(spacing: 5) {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 12.5))
                Text("Review").font(.system(size: 12.5, weight: .semibold))
            }
            .foregroundStyle(.orange)
        }
    }
}
