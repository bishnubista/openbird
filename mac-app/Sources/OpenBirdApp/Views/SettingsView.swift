import AppKit
import SwiftUI

/// The Settings pane (design: "Settings · Permissions, capture, and privacy"). The durable
/// configuration surface reached from the sidebar — a sticky header + a scroll of section
/// cards: a setup/status banner, Permissions & route, Capture allowlist (real app icons +
/// names), Privacy & storage, and Trust controls.
///
/// Every control binds to REAL `AppModel` state. Where the design shows copy the app can't
/// honestly back — the model names, the encryption "Enable" action — this view derives the
/// truth from state instead of hardcoding (see `modelRouteSubtitle`, the encryption row).
struct SettingsView: View {
    @ObservedObject var model: AppModel
    @Environment(\.colorScheme) private var scheme
    @State private var newBundleID = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: OB.Space.l) {            // 18 between sections
                    if let banner = bannerState { SettingsBanner(state: banner) }
                    permissionsCard
                    allowlistCard
                    privacyCard
                    promptCustomizationCard
                    trustCard
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 1) {
                Text("Settings")
                    .font(.system(size: 21, weight: .bold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(headerSubtitle)
                    .font(.system(size: 12.5))
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            Spacer()
            Button { Task { await model.refresh() } } label: {
                HStack(spacing: 7) {
                    Image(systemName: "arrow.clockwise").font(.system(size: 12, weight: .semibold))
                    Text("Re-check").font(.system(size: 13, weight: .medium))
                }
                .foregroundStyle(OB.textPrimary(scheme))
                .padding(.horizontal, 13)
                .padding(.vertical, 7)
                .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .disabled(model.isRefreshing || model.provisioningModel != nil)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .overlay(alignment: .bottom) {
            Rectangle().fill(OB.separator(scheme)).frame(height: 0.5)
        }
    }

    private var headerSubtitle: String {
        if let last = model.lastRefresh {
            return "Permissions, capture, and privacy · last checked \(last.formatted(date: .omitted, time: .shortened))"
        }
        return "Permissions, capture, and privacy"
    }

    // MARK: Setup / status banner

    private var bannerState: BannerState? {
        // 1) Recovery first: capture died and the index needs rebuilding.
        if model.captureNeedsReindex {
            return BannerState(
                tone: .amber,
                title: "Reindex needed to resume capture",
                subtitle: "The memory index must be rebuilt for the current embedding model.",
                buttonLabel: model.isReindexing ? "Reindexing…" : "Reindex now",
                action: model.isReindexing ? nil : { model.reindexNow() }
            )
        }
        // 2) Missing setup steps, in dependency order.
        if model.accessibilityState != .ok {
            return BannerState(
                tone: .amber,
                title: "Finish setup to start capturing",
                subtitle: "Grant Accessibility so OpenBird can read active-window text.",
                buttonLabel: "Grant Accessibility",
                action: { model.requestAccessibility() }
            )
        }
        if model.localModelStatusState != .ok {
            return BannerState(
                tone: .amber,
                title: "Finish setup to start capturing",
                subtitle: model.localModelStatusSummary,
                buttonLabel: model.modelRouteActionLabel,
                action: model.modelRouteActionLabel == nil ? nil : { model.performModelRouteAction { NSWorkspace.shared.open($0) } }
            )
        }
        if model.allowlist.isEmpty {
            return BannerState(
                tone: .amber,
                title: "Add an app to start capturing",
                subtitle: "OpenBird records nothing until you allow at least one app below.",
                buttonLabel: nil,
                action: nil
            )
        }
        if !model.captureRunning {
            return BannerState(
                tone: .amber,
                title: "Ready to capture",
                subtitle: "Start capturing to begin building your on-device memory.",
                buttonLabel: "Start capturing",
                action: { model.startCapture() }
            )
        }
        // 3) Everything's set — green.
        return BannerState(
            tone: .green,
            title: "\(model.capturePaused ? "Paused" : "Capturing") · everything's set",
            subtitle: allGoodSubtitle,
            buttonLabel: nil,
            action: nil
        )
    }

    private var allGoodSubtitle: String {
        let n = model.allowlist.count
        let apps = "\(n) app\(n == 1 ? "" : "s") allowed"
        let enc = model.encryptionState == .ok ? "encrypted" : "plaintext"
        // `modelRouteFooterLabel` already reads as a phrase ("on-device" / "remote model"),
        // so don't prefix "running on" — that would render "running on on-device".
        return "\(apps) · \(enc) · \(model.modelRouteFooterLabel)"
    }

    // MARK: Permissions & route

    private var permissionsCard: some View {
        sectionCard {
            sectionLabel("Permissions & route")
            permissionRow(
                icon: "terminal", title: "Active model route", badge: nil,
                subtitle: modelRouteSubtitle,
                trailing: model.localModelStatusState == .ok
                    ? .affirm("Verified")
                    : (model.modelRouteActionLabel.map { label in
                        RowTrailing.primary(label) { model.performModelRouteAction { NSWorkspace.shared.open($0) } }
                      } ?? .empty)
            )
            hairline
            permissionRow(
                icon: "accessibility", title: "Accessibility", badge: .required,
                subtitle: "Reads active-window text · approve OpenBird in the macOS prompt",
                trailing: model.accessibilityState == .ok
                    ? .affirm("Granted")
                    : .primary("Grant") { model.requestAccessibility() }
            )
            hairline
            permissionRow(
                icon: "rectangle.inset.filled", title: "Screen Recording", badge: .optional,
                subtitle: "Only needed for meeting / system-audio capture",
                trailing: model.screenRecordingState == .ok
                    ? .affirm("On")
                    : .secondary("Enable") { model.requestScreenRecording() }
            )
            hairline
            permissionRow(
                icon: "mic", title: "Microphone", badge: .optional,
                subtitle: "Records your side of a meeting as a separate track",
                trailing: model.microphoneState == .ok
                    ? .affirm("Granted")
                    : .secondary("Enable") { model.requestMicrophone() }
            )
            hairline
            permissionRow(
                icon: "waveform", title: "Meeting transcription", badge: .optional,
                subtitle: model.meetingTranscriptionSummary,
                trailing: meetingTranscriptionTrailing
            )
        }
    }

    /// Truthful model-route subtitle: the design's "qwen3:8b + embeddinggemma" is whatever
    /// the preflight actually requires (`requiredModelSummary`), never hardcoded; a remote
    /// or unverified route falls back to the honest status summary.
    private var modelRouteSubtitle: String {
        if model.localModelStatusState == .ok && !model.hasRemoteModelRoute {
            // Display join uses " + " (design); the underlying data keeps its own join.
            let models = model.requiredModelSummary.replacingOccurrences(of: ", ", with: " + ")
            return "Local Ollama · \(models)"
        }
        return model.localModelStatusSummary
    }

    private var meetingTranscriptionTrailing: RowTrailing {
        switch model.meetingTranscriptionState {
        case .ok:
            return .affirm("Verified")
        case .attention:
            return .attention("Needs install")
        case .unknown, .working:
            return .empty
        }
    }

    // MARK: Capture allowlist

    private var allowlistCard: some View {
        sectionCard {
            HStack(spacing: 9) {
                Text("Capture allowlist".uppercased())
                    .font(.system(size: 10.5, weight: .bold)).tracking(0.5)
                    .foregroundStyle(OB.textTertiary(scheme))
                Text("\(model.allowlist.count)")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(OB.accent)
                    .padding(.horizontal, 8).padding(.vertical, 1)
                    .background(OB.accent.opacity(0.13), in: Capsule())
                Spacer()
                Text("Text is captured only from these apps. Nothing else is read.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            .padding(.horizontal, 16).padding(.top, 13).padding(.bottom, 11)

            if model.allowlist.isEmpty {
                hairline
                Text("No apps allowed yet — capture will record nothing.")
                    .font(.system(size: 12))
                    .foregroundStyle(OB.amber)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            } else {
                ForEach(model.allowlist, id: \.self) { bundleID in
                    hairline
                    allowlistRow(bundleID)
                }
            }
            hairline
            addRow
            suggestions
        }
    }

    private func allowlistRow(_ bundleID: String) -> some View {
        let id = AppIdentity.forBundleID(bundleID)
        return HStack(spacing: 12) {
            appIconTile(id)
            VStack(alignment: .leading, spacing: 1) {
                Text(id.name)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(bundleID)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(OB.textTertiary(scheme))
            }
            Spacer()
            Button { model.removeFromAllowlist(bundleID) } label: {
                Image(systemName: "minus")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(OB.textTertiary(scheme))
                    .frame(width: 26, height: 26)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16).padding(.vertical, 9)
    }

    private func appIconTile(_ id: AppIdentity.Identity) -> some View {
        ZStack {
            if let icon = id.icon {
                Image(nsImage: icon).resizable().interpolation(.high).frame(width: 28, height: 28)
            } else {
                RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous).fill(OB.fieldFill(scheme))
                Image(systemName: "app.dashed")
                    .font(.system(size: 13))
                    .foregroundStyle(OB.textSecondary(scheme))
            }
        }
        .frame(width: 30, height: 30)
    }

    private var addRow: some View {
        HStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "plus").font(.system(size: 12)).foregroundStyle(OB.textTertiary(scheme))
                TextField("Add a bundle id (e.g. com.tinyspeck.slackmacgap)", text: $newBundleID)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .onSubmit(addEntry)
            }
            .padding(.horizontal, 12).padding(.vertical, 9)
            .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))

            Button(action: addEntry) {
                Text("Add")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .padding(.horizontal, 18).padding(.vertical, 9)
                    .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .disabled(newBundleID.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(.horizontal, 16).padding(.vertical, 12)
    }

    @ViewBuilder
    private var suggestions: some View {
        let sugg = model.runningAppSuggestions()
        if !sugg.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Running now — tap to allow")
                    .font(.system(size: 11)).foregroundStyle(OB.textTertiary(scheme))
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 7) {
                        ForEach(sugg.prefix(12), id: \.self) { suggestionChip($0) }
                    }
                }
            }
            .padding(.horizontal, 16).padding(.bottom, 14)
        }
    }

    private func suggestionChip(_ bundleID: String) -> some View {
        let id = AppIdentity.forBundleID(bundleID)
        return Button { model.addToAllowlist(bundleID) } label: {
            HStack(spacing: 7) {
                if let icon = id.icon {
                    Image(nsImage: icon).resizable().frame(width: 16, height: 16)
                } else {
                    Image(systemName: "app.dashed").font(.system(size: 11)).foregroundStyle(OB.textSecondary(scheme))
                }
                Text(id.name).font(.system(size: 12, weight: .medium)).foregroundStyle(OB.textPrimary(scheme))
                Image(systemName: "plus").font(.system(size: 10)).foregroundStyle(OB.textTertiary(scheme))
            }
            .padding(.leading, 7).padding(.trailing, 11).padding(.vertical, 5)
            .background(OB.fieldFill(scheme), in: Capsule())
            .overlay(Capsule().strokeBorder(OB.separator(scheme), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    // MARK: Privacy & storage

    private var privacyCard: some View {
        sectionCard {
            sectionLabel("Privacy & storage")
            permissionRow(
                icon: "lock", title: "Encrypted memory", badge: nil,
                subtitle: encryptionDetail,
                // No in-app enable path exists (it requires the SQLCipher/Homebrew build), so
                // this stays an honest status indicator — never a button that does nothing.
                trailing: model.encryptionState == .ok
                    ? .affirm("On")
                    : .attention(model.encryptionState == .attention ? "Plaintext" : "Unknown")
            )
            hairline
            permissionRow(
                icon: "folder", title: "Local data", badge: nil,
                subtitle: "\(model.memoryStats.observations) \(model.memoryStats.observations == 1 ? "observation" : "observations") stored on this Mac",
                trailing: .secondary("Reveal") { model.openDataFolder() }
            )
            hairline
            HStack(spacing: 8) {
                Image(systemName: "lock.fill").font(.system(size: 11))
                Text(model.privacyTransmissionSummary)
                    .font(.system(size: 11.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(OB.textTertiary(scheme))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16).padding(.vertical, 11)
        }
    }

    private var encryptionDetail: String {
        switch model.encryptionState {
        case .ok: return "At-rest SQLCipher encryption is active"
        case .attention: return "Running plaintext — enable SQLCipher (bundled with the Homebrew build) to encrypt stored text"
        default: return "Encryption status could not be verified"
        }
    }

    // MARK: Prompt customization

    private var promptCustomizationCard: some View {
        sectionCard {
            sectionLabel("Prompt customization")
            permissionRow(
                icon: "text.quote",
                title: "Effective prompts directory",
                badge: nil,
                subtitle: model.promptDirectoryPath,
                trailing: .secondary("Open") { model.openPromptsFolder() }
            )
            hairline
            VStack(alignment: .leading, spacing: 9) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Persona overrides")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    Text("Edit the persona files that wrap Ask, routine, meeting, and signal prompts.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(OB.textSecondary(scheme))
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack(spacing: 8) {
                    ForEach(PromptPersonaKey.allCases) { key in
                        promptEditButton(key)
                    }
                    Spacer(minLength: 0)
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
            hairline
            HStack(spacing: 8) {
                Image(systemName: "lock.shield").font(.system(size: 11))
                Text("OpenBird keeps the security scaffold app-owned; these files change only the persona.")
                    .font(.system(size: 11.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(OB.textTertiary(scheme))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16).padding(.vertical, 11)
        }
    }

    private func promptEditButton(_ key: PromptPersonaKey) -> some View {
        Button { model.editPromptPersona(key) } label: {
            HStack(spacing: 6) {
                Image(systemName: "square.and.pencil")
                    .font(.system(size: 11, weight: .semibold))
                Text(key.label)
                    .font(.system(size: 12.5, weight: .semibold))
            }
            .foregroundStyle(OB.textPrimary(scheme))
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    // MARK: Trust controls

    private var trustCard: some View {
        sectionCard {
            sectionLabel("Trust controls")
            HStack(spacing: 13) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Pause capture")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    Text("Temporarily stop reading window text — resume anytime")
                        .font(.system(size: 11.5))
                        .foregroundStyle(OB.textSecondary(scheme))
                }
                Spacer()
                PauseToggle(isOn: model.capturePaused) { model.toggleCapturePause() }
            }
            .padding(.horizontal, 16).padding(.vertical, 11)
            hairline
            HStack(spacing: 10) {
                trustButton("Stop helpers", system: "stop.fill") { model.stopHelpers() }
                trustButton("App bundle", system: "app") { model.openBundleFolder() }
                Spacer()
                HStack(spacing: 14) {
                    ForEach(model.helpers) { helper in
                        HStack(spacing: 5) {
                            Circle()
                                .fill(helper.isBundled ? OB.ok(scheme) : Color.secondary)
                                .frame(width: 6, height: 6)
                            Text(helper.label)
                                .font(.system(size: 11.5))
                                .foregroundStyle(OB.textSecondary(scheme))
                        }
                    }
                }
            }
            .padding(.horizontal, 16).padding(.vertical, 11)
        }
    }

    private func trustButton(_ label: String, system: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: system).font(.system(size: 11, weight: .semibold))
                Text(label).font(.system(size: 12.5, weight: .semibold))
            }
            .foregroundStyle(OB.textPrimary(scheme))
            .padding(.horizontal, 13).padding(.vertical, 7)
            .background(OB.fieldFill(scheme), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private func addEntry() {
        // `.onSubmit` bypasses the Add button's disabled guard, so trim/guard here too —
        // don't no-op-clear the field on an empty/whitespace submit. (`addToAllowlist`
        // also guards, but keeping the intent explicit avoids a confusing field reset.)
        let id = newBundleID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else { return }
        model.addToAllowlist(id)
        newBundleID = ""
    }

    // MARK: Shared section primitives

    private func sectionCard<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 0) { content() }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(OB.cardFill(scheme))
            .clipShape(RoundedRectangle(cornerRadius: OB.Radius.setup, style: .continuous))   // 14
            .overlay(RoundedRectangle(cornerRadius: OB.Radius.setup, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 10.5, weight: .bold)).tracking(0.5)
            .foregroundStyle(OB.textTertiary(scheme))
            .padding(.top, 13).padding(.bottom, 10).padding(.horizontal, 16)
    }

    private var hairline: some View {
        Rectangle().fill(OB.separator(scheme)).frame(height: 0.5)
    }

    private func permissionRow(icon: String, title: String, badge: RowBadge?, subtitle: String, trailing: RowTrailing) -> some View {
        HStack(spacing: 13) {
            iconTile(icon)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text(title)
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    if let badge { badgeView(badge) }
                }
                Text(subtitle)
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: OB.Space.sm)
            trailingView(trailing)
        }
        .padding(.horizontal, 16).padding(.vertical, 11)
    }

    private func iconTile(_ icon: String) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: OB.Radius.tile, style: .continuous).fill(OB.accent.opacity(0.14))
            Image(systemName: icon).font(.system(size: 16)).foregroundStyle(OB.accent)
        }
        .frame(width: 32, height: 32)
    }

    private func badgeView(_ badge: RowBadge) -> some View {
        let isRequired = badge == .required
        return Text(isRequired ? "Required" : "Optional")
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(isRequired ? OB.accent : OB.textTertiary(scheme))
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(
                isRequired ? OB.accent.opacity(0.13) : OB.fieldFill(scheme),
                in: RoundedRectangle(cornerRadius: 5, style: .continuous)
            )
    }

    @ViewBuilder
    private func trailingView(_ trailing: RowTrailing) -> some View {
        switch trailing {
        case .affirm(let label):
            HStack(spacing: 5) {
                Image(systemName: "checkmark").font(.system(size: 12.5, weight: .bold))
                Text(label).font(.system(size: 12.5, weight: .semibold))
            }
            .foregroundStyle(OB.ok(scheme))
        case .attention(let label):
            HStack(spacing: 5) {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 11.5))
                Text(label).font(.system(size: 12.5, weight: .semibold))
            }
            .foregroundStyle(OB.amber)
        case .primary(let label, let action):
            Button(action: action) {
                Text(label)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 15).padding(.vertical, 7)
                    .background(OB.accent, in: RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous))
            }
            .buttonStyle(.plain)
        case .secondary(let label, let action):
            Button(action: action) {
                Text(label)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                    .padding(.horizontal, 15).padding(.vertical, 7)
                    .overlay(RoundedRectangle(cornerRadius: OB.Radius.control, style: .continuous).strokeBorder(OB.separator(scheme), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
        case .empty:
            EmptyView()
        }
    }
}

// MARK: - Row models

private enum RowBadge { case required, optional }

private enum RowTrailing {
    case affirm(String)
    case attention(String)
    case primary(String, () -> Void)
    case secondary(String, () -> Void)
    case empty
}

private struct BannerState {
    enum Tone { case amber, green }
    let tone: Tone
    let title: String
    let subtitle: String
    let buttonLabel: String?
    let action: (() -> Void)?
}

// MARK: - Banner

private struct SettingsBanner: View {
    let state: BannerState
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let tint = state.tone == .amber ? OB.amber : OB.ok(scheme)
        return HStack(spacing: OB.Space.ml) {
            ZStack {
                RoundedRectangle(cornerRadius: 11, style: .continuous).fill(tint.opacity(0.18))
                Image(systemName: state.tone == .amber ? "exclamationmark.circle" : "checkmark")
                    .font(.system(size: 18, weight: state.tone == .amber ? .semibold : .bold))
                    .foregroundStyle(tint)
            }
            .frame(width: 38, height: 38)

            VStack(alignment: .leading, spacing: 1) {
                Text(state.title)
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(state.subtitle)
                    .font(.system(size: 12.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: OB.Space.m)

            if let label = state.buttonLabel {
                Button { state.action?() } label: {
                    Text(label)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, OB.Space.ml).padding(.vertical, 9)
                        .background(OB.accent, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(state.action == nil)
            }
        }
        .padding(.horizontal, 18).padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: OB.Radius.setup, style: .continuous)
                .fill(LinearGradient(
                    colors: [tint.opacity(0.14), tint.opacity(0.05)],
                    startPoint: .top, endPoint: .bottom
                ))
        )
        .overlay(
            RoundedRectangle(cornerRadius: OB.Radius.setup, style: .continuous)
                .strokeBorder(tint.opacity(0.35), lineWidth: 0.5)
        )
    }
}

// MARK: - Pause toggle

/// The design's 40×24 pill toggle — amber while paused, neutral while capture is live.
private struct PauseToggle: View {
    let isOn: Bool
    let action: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        Button(action: action) {
            ZStack(alignment: isOn ? .trailing : .leading) {
                Capsule().fill(isOn ? OB.amber : OB.separator(scheme))
                    .frame(width: 40, height: 24)
                Circle().fill(.white)
                    .frame(width: 20, height: 20)
                    .padding(2)
                    .shadow(color: .black.opacity(0.3), radius: 1, y: 1)
            }
            .frame(width: 40, height: 24)
            .animation(.easeInOut(duration: 0.15), value: isOn)
        }
        .buttonStyle(.plain)
    }
}
