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
    @State private var pendingDetailedCaptureBundleID: String?
    @State private var pendingClaudeConnection = false
    @State private var pendingChatGPTConnection = false
    @State private var chatGPTTunnelID = ""
    @State private var chatGPTRuntimeKey = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: OB.Space.l) {            // 18 between sections
                    if let banner = bannerState { SettingsBanner(state: banner) }
                    if !model.lastActionMessage.isEmpty {
                        SettingsActionMessage(message: model.lastActionMessage)
                    }
                    permissionsCard
                    allowlistCard
                    detailedCaptureCard
                    deepCaptureCard
                    privacyCard
                    assistantCard
                    promptCustomizationCard
                    trustCard
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .alert(
            "Enable detailed local capture?",
            isPresented: Binding(
                get: { pendingDetailedCaptureBundleID != nil },
                set: { if !$0 { pendingDetailedCaptureBundleID = nil } }
            )
        ) {
            Button("Cancel", role: .cancel) {
                pendingDetailedCaptureBundleID = nil
            }
            Button("Enable") {
                if let bundleID = pendingDetailedCaptureBundleID {
                    model.setDetailedCapture(bundleID, enabled: true)
                }
                pendingDetailedCaptureBundleID = nil
            }
        } message: {
            Text(
                "Terminal and editor text can contain commands, output, environment values, "
                + "and copied secrets. OpenBird will store readable text in encrypted local "
                + "memory; secure fields, private windows, and password managers stay blocked."
            )
        }
        .alert(AssistantConsentCopy.claudeConnectTitle, isPresented: $pendingClaudeConnection) {
            Button("Cancel", role: .cancel) {}
            Button("Connect") { model.connectClaudeAssistant() }
        } message: {
            Text(AssistantConsentCopy.claudeConnectMessage)
        }
        .sheet(
            isPresented: $pendingChatGPTConnection,
            onDismiss: { chatGPTRuntimeKey = "" }
        ) { chatGPTSetupSheet }
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
        let decision = SettingsBannerDecision.resolve(
            captureNeedsReindex: model.captureNeedsReindex,
            isReindexing: model.isReindexing,
            accessibilityState: model.accessibilityState,
            localModelStatusState: model.localModelStatusState,
            localModelStatusSummary: model.localModelStatusSummary,
            modelRouteActionLabel: model.modelRouteActionLabel,
            allowlistCount: model.allowlist.count,
            captureRunning: model.captureRunning,
            capturePaused: model.capturePaused,
            observationCount: model.memoryStats.observations,
            memoryStatsState: model.memoryStatsState,
            isRefreshing: model.isRefreshing,
            nextStepSummary: model.nextStepSummary,
            allGoodSubtitle: allGoodSubtitle
        )
        return BannerState(decision: decision, action: bannerAction(for: decision.actionKind))
    }

    private func bannerAction(for kind: SettingsBannerActionKind) -> (() -> Void)? {
        switch kind {
        case .none:
            return nil
        case .reindex:
            return model.isReindexing ? nil : { model.reindexNow() }
        case .requestAccessibility:
            return { model.requestAccessibility() }
        case .modelRoute:
            guard model.modelRouteActionLabel != nil else { return nil }
            return { model.performModelRouteAction { NSWorkspace.shared.open($0) } }
        case .startCapture:
            return { model.startCapture() }
        }
    }

    private var allGoodSubtitle: String {
        let enc = model.encryptionState == .ok ? "encrypted" : "plaintext"
        // `modelRouteFooterLabel` already reads as a phrase ("on-device" / "remote model"),
        // so don't prefix "running on" — that would render "running on on-device".
        return "\(model.captureAllowedSummary) · \(enc) · \(model.modelRouteFooterLabel)"
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
                subtitle: "Needed for meetings and OCR fallback in opted-in apps",
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
                Text("Active-window text is captured only from effectively allowed apps; safety blocks can override this list.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .multilineTextAlignment(.trailing)
                    .fixedSize(horizontal: false, vertical: true)
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
        let status = model.captureRowStatus(for: bundleID)
        return HStack(spacing: 12) {
            appIconTile(id)
            VStack(alignment: .leading, spacing: 1) {
                Text(id.name)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(bundleID)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(OB.textTertiary(scheme))
                    .lineLimit(1)
                Text(status.detail)
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Spacer()
            captureStatusPill(status)
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

    private func captureStatusPill(_ status: CaptureRowStatus) -> some View {
        let colors = captureStatusColors(status.tone)
        return Text(status.label)
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(colors.foreground)
            .lineLimit(1)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(colors.background, in: Capsule())
            .overlay(Capsule().strokeBorder(colors.border, lineWidth: 0.5))
            .fixedSize()
    }

    private func captureStatusColors(_ tone: CaptureRowTone) -> (foreground: Color, background: Color, border: Color) {
        switch tone {
        case .ok:
            let green = OB.ok(scheme)
            return (green, green.opacity(0.13), green.opacity(0.28))
        case .attention:
            return (OB.amber, OB.amber.opacity(0.13), OB.amber.opacity(0.3))
        case .neutral:
            return (OB.textSecondary(scheme), OB.fieldFill(scheme), OB.separator(scheme))
        }
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

    // MARK: Detailed local capture

    private var detailedCaptureCard: some View {
        sectionCard {
            sectionLabel("Detailed local capture")
            Text(
                "Terminals and editors are blocked by default because their visible text can "
                + "include secrets. Enable only the apps whose work context you want OpenBird "
                + "to remember. Verified encrypted memory is required."
            )
            .font(.system(size: 11.5))
            .foregroundStyle(OB.textSecondary(scheme))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 16).padding(.bottom, 12)

            if model.detailedCaptureEligibleApps.isEmpty {
                hairline
                Text("Allowlisted terminals and editors appear here after capture health is checked.")
                    .font(.system(size: 12))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            } else {
                ForEach(model.detailedCaptureEligibleApps, id: \.self) { bundleID in
                    hairline
                    detailedCaptureRow(bundleID)
                }
            }
        }
    }

    private func detailedCaptureRow(_ bundleID: String) -> some View {
        let id = AppIdentity.forBundleID(bundleID)
        let enabled = model.detailedCaptureApps.contains(bundleID)
        return HStack(spacing: 12) {
            appIconTile(id)
            VStack(alignment: .leading, spacing: 1) {
                Text(id.name)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(OB.textPrimary(scheme))
                Text(enabled ? "Detailed text capture enabled" : "Blocked by default")
                    .font(.system(size: 11.5))
                    .foregroundStyle(OB.textSecondary(scheme))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Toggle(
                "Detailed capture for \(id.name)",
                isOn: Binding(
                    get: { model.detailedCaptureApps.contains(bundleID) },
                    set: { value in
                        if value {
                            pendingDetailedCaptureBundleID = bundleID
                        } else {
                            model.setDetailedCapture(bundleID, enabled: false)
                        }
                    }
                )
            )
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
            .disabled(model.encryptionState != .ok && !enabled)
        }
        .padding(.horizontal, 16).padding(.vertical, 9)
    }

    // MARK: Deep capture (OCR) — Phase C2 opt-in

    /// The truth surface `privacy-routes.yaml` names `app.deep_capture_section`.
    /// Per-app toggles are rendered ONLY for allowlisted apps (the structural
    /// subset, mirrored in the UI), and the copy states the three real costs
    /// of the Screen Recording route up front — never a silent grant.
    private var deepCaptureCard: some View {
        sectionCard {
            sectionLabel("Deep capture (OCR)")
            Text(
                "When an allowed app exposes no readable text, OpenBird can take a "
                + "window-scoped screenshot and read it with on-device OCR. Pixels are "
                + "discarded immediately; only recognized text is stored, scrubbed like "
                + "any other capture. Costs: the Screen Recording permission, macOS's "
                + "monthly re-auth nag, and a permanent orange indicator in the menu bar."
            )
            .font(.system(size: 11.5))
            .foregroundStyle(OB.textSecondary(scheme))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 16).padding(.bottom, 12)

            hairline
            permissionRow(
                icon: "rectangle.inset.filled", title: "Screen Recording", badge: .required,
                subtitle: "Required before any deep capture can run · granted to the OpenBird app",
                trailing: model.screenRecordingGranted
                    ? .affirm("Granted")
                    : .primary("Grant") { model.requestScreenRecording() }
            )

            if model.allowlist.isEmpty {
                hairline
                Text("Add apps to the capture allowlist first — deep capture is a per-app upgrade, never a blanket grant.")
                    .font(.system(size: 12))
                    .foregroundStyle(OB.textSecondary(scheme))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16).padding(.vertical, 12)
            } else {
                // Toggles ONLY for allowlisted apps: opted-in ⊆ allowlist.
                ForEach(model.allowlist, id: \.self) { bundleID in
                    hairline
                    deepCaptureRow(bundleID)
                }
            }
        }
    }

    private func deepCaptureRow(_ bundleID: String) -> some View {
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
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Spacer()
            Toggle(
                "Deep capture for \(id.name)",
                isOn: Binding(
                    get: { model.ocrApps.contains(bundleID) },
                    set: { model.setOcrCapture(bundleID, enabled: $0) }
                )
            )
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)
            // The toggle is inert until Screen Recording is actually granted:
            // an opt-in the helper cannot honor would be a silent lie.
            .disabled(!model.screenRecordingGranted)
        }
        .padding(.horizontal, 16).padding(.vertical, 9)
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
            permissionRow(
                icon: "square.and.arrow.up",
                title: "Export memory (Terminal)",
                badge: nil,
                subtitle: model.dataExportSummary,
                trailing: .secondary("Copy export command") { model.copyDataExportCommand() }
            )
            hairline
            permissionRow(
                icon: "trash",
                title: "Delete memory (Terminal)",
                badge: nil,
                subtitle: model.dataDeletionSummary,
                trailing: .secondary("Copy prune command") { model.copyDataPruneCommand() }
            )
            hairline
            permissionRow(
                icon: "brain.head.profile",
                title: "Deep Brain",
                badge: .optional,
                subtitle: model.deepBrainStatusSummary,
                trailing: model.deepBrainStatusNeedsAttention
                    ? .attention(model.deepBrainStatusBadge)
                    : .affirm(model.deepBrainStatusBadge)
            )
            hairline
            permissionRow(
                icon: "doc.text.magnifyingglass",
                title: "Deep Brain packet preview",
                badge: nil,
                subtitle: model.deepBrainPreviewSummary,
                trailing: deepBrainPreviewTrailing
            )
            hairline
            permissionRow(
                icon: "terminal",
                title: "Deep Brain ask (Terminal)",
                badge: nil,
                subtitle: model.deepBrainAskCommandSummary,
                trailing: .secondary("Copy ask command") { model.copyDeepBrainAskCommand() }
            )
            hairline
            permissionRow(
                icon: "terminal",
                title: "Productivity coach (Terminal)",
                badge: nil,
                subtitle: model.productivityCoachCommandSummary,
                trailing: .secondary("Copy coach command") { model.copyProductivityCoachCommand() }
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

    private var deepBrainPreviewTrailing: RowTrailing {
        switch model.deepBrainPreviewState {
        case .loading:
            return .attention("Loading")
        case .loaded:
            return .affirm(model.deepBrainPreviewBadge)
        case .unknown, .failed:
            return .secondary(model.deepBrainPreviewBadge) { model.loadDeepBrainPreview() }
        }
    }

    // MARK: Desktop assistants

    private var assistantCard: some View {
        sectionCard {
            sectionLabel("Desktop assistants")
            permissionRow(
                icon: "sparkles",
                title: "Claude Desktop",
                badge: .optional,
                subtitle: model.claudeAssistantSummary,
                trailing: claudeAssistantTrailing
            )
            hairline
            permissionRow(
                icon: "bubble.left.and.text.bubble.right",
                title: "ChatGPT",
                badge: .optional,
                subtitle: model.chatGPTAssistantSummary,
                trailing: chatGPTAssistantTrailing
            )
            hairline
            HStack(spacing: 8) {
                Image(systemName: "lock.shield").font(.system(size: 11))
                Text("URLs and window titles stay local. Existing outbound exclusions apply.")
                    .font(.system(size: 11.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(OB.textTertiary(scheme))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16).padding(.vertical, 11)
        }
    }

    private var claudeAssistantTrailing: RowTrailing {
        if model.claudeAssistantBusy { return .attention("Connecting") }
        switch model.claudeAssistantState {
        case .connected:
            return .affirm("Connected")
        case .unknown:
            return .attention("Check")
        case .disconnected, .failed:
            return .primary("Connect") { pendingClaudeConnection = true }
        }
    }

    private var chatGPTAssistantTrailing: RowTrailing {
        switch model.chatGPTAssistantState {
        case .connected:
            return .affirmAction("Connected") { pendingChatGPTConnection = true }
        case .connecting:
            return .attention("Connecting")
        case .unknown:
            return .attention("Check")
        case .setupNeeded, .needsAttention:
            return .primary("Connect") { pendingChatGPTConnection = true }
        }
    }

    private var chatGPTSetupSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connect ChatGPT")
                .font(.system(size: 20, weight: .bold))
            Text(AssistantConsentCopy.chatGPTConnectMessage)
            .font(.system(size: 12.5))
            .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 8) {
                Button("1. Open ChatGPT Developer Mode settings") {
                    NSWorkspace.shared.open(URL(string: "https://chatgpt.com/#settings/Connectors")!)
                }
                Button("2. Create or select a Secure MCP Tunnel") {
                    NSWorkspace.shared.open(URL(string: "https://platform.openai.com/settings/organization/tunnels")!)
                }
            }
            TextField("Tunnel id (tunnel_...)", text: $chatGPTTunnelID)
                .textFieldStyle(.roundedBorder)
            SecureField("Restricted tunnel runtime API key", text: $chatGPTRuntimeKey)
                .textFieldStyle(.roundedBorder)
            Text("The runtime key is stored in your Mac Keychain and never placed in logs or command arguments.")
                .font(.system(size: 11.5))
                .foregroundStyle(.secondary)
            HStack {
                if model.chatGPTAssistantState == .connected {
                    Button("Remove connection", role: .destructive) {
                        model.removeChatGPTAssistant()
                        pendingChatGPTConnection = false
                    }
                }
                Spacer()
                Button("Cancel", role: .cancel) {
                    chatGPTRuntimeKey = ""
                    pendingChatGPTConnection = false
                }
                Button("Connect") {
                    model.connectChatGPTAssistant(
                        tunnelID: chatGPTTunnelID.trimmingCharacters(in: .whitespacesAndNewlines),
                        runtimeKey: chatGPTRuntimeKey
                    )
                    chatGPTRuntimeKey = ""
                    pendingChatGPTConnection = false
                }
                .disabled(
                    !chatGPTTunnelID.hasPrefix("tunnel_") || chatGPTRuntimeKey.isEmpty
                    || model.chatGPTAssistantState == .connecting
                )
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 520)
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
            HStack(spacing: 13) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Launch at login")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(OB.textPrimary(scheme))
                    Text("Start OpenBird at login so memory keeps building after a restart")
                        .font(.system(size: 11.5))
                        .foregroundStyle(OB.textSecondary(scheme))
                }
                Spacer()
                PauseToggle(isOn: model.launchAtLogin) { model.setLaunchAtLogin(!model.launchAtLogin) }
            }
            .padding(.horizontal, 16).padding(.vertical, 11)
            hairline
            HStack(spacing: 10) {
                trustButton("Stop capture helper", system: "stop.fill") { model.stopHelpers() }
                trustButton("Force stop meeting audio", system: "waveform.slash") {
                    model.forceStopMeetingAudio()
                }
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
        case .affirmAction(let label, let action):
            Button(action: action) {
                HStack(spacing: 5) {
                    Image(systemName: "checkmark").font(.system(size: 12.5, weight: .bold))
                    Text(label).font(.system(size: 12.5, weight: .semibold))
                }
                .foregroundStyle(OB.ok(scheme))
            }
            .buttonStyle(.plain)
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
    case affirmAction(String, () -> Void)
    case attention(String)
    case primary(String, () -> Void)
    case secondary(String, () -> Void)
    case empty
}

enum SettingsBannerTone: Equatable {
    case amber
    case green
}

enum SettingsBannerActionKind: Equatable {
    case none
    case reindex
    case requestAccessibility
    case modelRoute
    case startCapture
}

struct SettingsBannerDecision: Equatable {
    let tone: SettingsBannerTone
    let title: String
    let subtitle: String
    let buttonLabel: String?
    let actionKind: SettingsBannerActionKind

    static func resolve(
        captureNeedsReindex: Bool,
        isReindexing: Bool,
        accessibilityState: StepState,
        localModelStatusState: StepState,
        localModelStatusSummary: String,
        modelRouteActionLabel: String?,
        allowlistCount: Int,
        captureRunning: Bool,
        capturePaused: Bool,
        observationCount: Int,
        memoryStatsState: MemoryStatsState,
        isRefreshing _: Bool,
        nextStepSummary: String,
        allGoodSubtitle: String
    ) -> SettingsBannerDecision {
        if captureNeedsReindex {
            return SettingsBannerDecision(
                tone: .amber,
                title: "Reindex needed to resume capture",
                subtitle: "The memory index must be rebuilt for the current embedding model.",
                buttonLabel: isReindexing ? "Reindexing…" : "Reindex now",
                actionKind: isReindexing ? .none : .reindex
            )
        }
        if accessibilityState != .ok {
            return SettingsBannerDecision(
                tone: .amber,
                title: "Finish setup to start capturing",
                subtitle: "Grant Accessibility so OpenBird can read active-window text.",
                buttonLabel: "Grant Accessibility",
                actionKind: .requestAccessibility
            )
        }
        if localModelStatusState != .ok {
            return SettingsBannerDecision(
                tone: .amber,
                title: "Finish setup to start capturing",
                subtitle: localModelStatusSummary,
                buttonLabel: modelRouteActionLabel,
                actionKind: modelRouteActionLabel == nil ? .none : .modelRoute
            )
        }
        if allowlistCount == 0 {
            return SettingsBannerDecision(
                tone: .amber,
                title: "Add an app to start capturing",
                subtitle: "OpenBird records nothing until you allow at least one app below.",
                buttonLabel: nil,
                actionKind: .none
            )
        }
        if !captureRunning {
            return SettingsBannerDecision(
                tone: .amber,
                title: "Ready to capture",
                subtitle: "Start capturing to begin building your on-device memory.",
                buttonLabel: "Start capturing",
                actionKind: .startCapture
            )
        }
        switch memoryStatsState {
        case .unknown:
            return SettingsBannerDecision(
                tone: .amber,
                title: "\(capturePaused ? "Paused" : "Capturing") · checking memory",
                subtitle: "OpenBird is verifying whether captured memory is available.",
                buttonLabel: nil,
                actionKind: .none
            )
        case .failed:
            return SettingsBannerDecision(
                tone: .amber,
                title: "Memory status unavailable",
                subtitle: "Re-check setup to verify captured memory.",
                buttonLabel: nil,
                actionKind: .none
            )
        case .loaded:
            break
        }
        if observationCount == 0 {
            let status = capturePaused ? "Paused" : "Capturing"
            let subtitle = capturePaused
                ? "Resume capture and bring an allowed app to the front so memory starts filling."
                : nextStepSummary
            return SettingsBannerDecision(
                tone: .amber,
                title: "\(status) · waiting for memory",
                subtitle: subtitle,
                buttonLabel: nil,
                actionKind: .none
            )
        }
        return SettingsBannerDecision(
            tone: .green,
            title: "\(capturePaused ? "Paused" : "Capturing") · memory ready",
            subtitle: allGoodSubtitle,
            buttonLabel: nil,
            actionKind: .none
        )
    }
}

private struct BannerState {
    let tone: SettingsBannerTone
    let title: String
    let subtitle: String
    let buttonLabel: String?
    let action: (() -> Void)?

    init(decision: SettingsBannerDecision, action: (() -> Void)?) {
        self.tone = decision.tone
        self.title = decision.title
        self.subtitle = decision.subtitle
        self.buttonLabel = decision.buttonLabel
        self.action = action
    }
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

private struct SettingsActionMessage: View {
    let message: String
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        HStack(spacing: OB.Space.s) {
            Image(systemName: "info.circle")
                .font(.system(size: 12.5, weight: .semibold))
            Text(message)
                .font(.system(size: 12.5))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .foregroundStyle(OB.textSecondary(scheme))
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(
            OB.fieldFill(scheme),
            in: RoundedRectangle(cornerRadius: OB.Radius.card, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: OB.Radius.card, style: .continuous)
                .strokeBorder(OB.separator(scheme), lineWidth: 0.5)
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
