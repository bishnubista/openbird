import SwiftUI

/// Quick chat over captured memory, with occurrence-level citations. Calls
/// `openbird chat --json` via the service (off the main actor) and renders the
/// grounded answer plus where each fact came from.
struct ChatView: View {
    @ObservedObject var model: AppModel
    @State private var question = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ask your memory")
                .font(.headline)

            Label(model.localModelStatusSummary, systemImage: statusIcon)
                .font(.caption)
                .foregroundStyle(statusColor)

            HStack {
                TextField("e.g. what did we decide about storage?", text: $question)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.ask(question) }
                Button {
                    model.ask(question)
                } label: {
                    if model.chatBusy {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Ask")
                    }
                }
                .disabled(model.chatBusy
                    || question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || model.askUnavailableReason != nil)
            }

            if let reason = model.askUnavailableReason {
                Label(reason, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let err = model.chatError {
                Label(err, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.red)
            }

            if let result = model.chatResult {
                if !result.grounded && !result.answer.isEmpty {
                    Label("ungrounded — no verified source for this answer",
                          systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                Text(result.answer.isEmpty ? "(no answer)" : result.answer)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)

                if !result.citations.isEmpty {
                    Text("Sources")
                        .font(.subheadline).bold()
                    ForEach(result.citations) { citation in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("[\(citation.index)] \(sourceLabel(citation)) · \(timeLabel(citation.ts))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(citation.snippet)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                                .lineLimit(3)
                        }
                    }
                }
            }
        }
    }

    private func sourceLabel(_ citation: ChatCitation) -> String {
        let parts = [citation.app, citation.window]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        return parts.isEmpty ? "unknown" : parts.joined(separator: " / ")
    }

    private func timeLabel(_ ts: Double) -> String {
        Date(timeIntervalSince1970: ts).formatted(date: .abbreviated, time: .shortened)
    }

    private var statusIcon: String {
        switch model.localModelStatusState {
        case .ok: return "checkmark.circle.fill"
        case .attention: return "exclamationmark.triangle.fill"
        case .working: return "arrow.clockwise.circle.fill"
        case .unknown: return "questionmark.circle"
        }
    }

    private var statusColor: Color {
        switch model.localModelStatusState {
        case .ok: return .green
        case .attention: return .orange
        case .working: return .blue
        case .unknown: return .secondary
        }
    }
}
