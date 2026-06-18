import AppKit
import Foundation

struct HelperStatus: Identifiable, Equatable {
    let id: String
    let label: String
    let isBundled: Bool
    let path: String
}

struct PreflightSummary: Equatable {
    let status: String
    let detail: String
}

@MainActor
final class AppModel: ObservableObject {
    @Published private(set) var capturePaused = false
    @Published private(set) var helpers: [HelperStatus] = []
    @Published private(set) var preflight = PreflightSummary(
        status: "Unknown",
        detail: "Refresh to inspect this OpenBird install."
    )
    @Published private(set) var lastRefresh: Date?
    @Published private(set) var isRefreshing = false
    @Published var lastActionMessage = ""

    private let service: OpenBirdService

    init(service: OpenBirdService) {
        self.service = service
        self.capturePaused = service.isCapturePaused()
        self.helpers = service.helperStatuses()
    }

    var menuBarSymbol: String {
        capturePaused ? "pause.circle" : "bird"
    }

    func refresh() async {
        isRefreshing = true
        defer { isRefreshing = false }

        capturePaused = service.isCapturePaused()
        helpers = service.helperStatuses()
        preflight = await service.preflightSummary()
        lastRefresh = Date()
    }

    func toggleCapturePause() {
        do {
            capturePaused = try service.setCapturePaused(!capturePaused)
            lastActionMessage = capturePaused ? "Capture paused." : "Capture resumed."
        } catch {
            lastActionMessage = "Could not update pause state: \(error.localizedDescription)"
        }
    }

    func stopHelpers() {
        let stopped = service.stopHelperProcesses()
        lastActionMessage = stopped
            ? "Stopped helper processes."
            : "No helper processes were running."
    }

    func openDataFolder() {
        service.openDataFolder()
    }

    func openBundleFolder() {
        service.openBundleFolder()
    }
}
