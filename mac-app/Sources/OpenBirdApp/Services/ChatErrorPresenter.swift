import Foundation

/// Single source of truth for turning a chat `Error` into friendly, content-safe
/// UI text. Used by both `AppModel.ask` (window chat) and `AskPanelModel.ask`
/// (Spotlight panel) so the two surfaces never drift in their error wording.
enum ChatErrorPresenter {
    static func describe(_ error: Error) -> String {
        switch error {
        case ChatError.cliMissing:
            return "OpenBird CLI not found in the app bundle."
        case ChatError.failed(let message):
            return message
        case ChatError.decode:
            return "Could not read the chat response."
        default:
            return "Chat error."
        }
    }
}
