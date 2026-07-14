import Foundation

/// The native assistant-connect consent copy, extracted so tests can pin it.
///
/// These strings are primary consent surfaces: the app's connect flows invoke
/// the CLI with `--yes`, bypassing its installer warning, so what the user
/// reads here is the ONLY disclosure they see. The privacy route manifest
/// (`docs/privacy-routes.yaml`) lists them as truth surfaces — every category
/// of data an assistant tool can egress (excerpts, app identifiers including
/// redacted apps with reason codes, activity patterns, store totals, host
/// label) must be named before the user confirms.
enum AssistantConsentCopy {
    static let claudeConnectTitle = "Connect Claude Desktop?"

    static let claudeConnectMessage =
        "Claude can request bounded OpenBird excerpts, app identifiers — including "
        + "apps whose content was redacted, with reason codes — timestamps, app usage "
        + "durations, activity patterns (focus, meetings, context switches), "
        + "memory-store totals, encryption state, exclusion counts, and a host label "
        + "identifying this Mac. Those results are sent to Anthropic and cannot be "
        + "recalled by deleting local memory. OpenBird never sends data in the "
        + "background."

    static let chatGPTConnectMessage =
        "ChatGPT connects to this Mac through OpenAI Secure MCP Tunnel. OpenBird "
        + "never uploads capture in the background; bounded excerpts, app identifiers "
        + "— including apps whose content was redacted, with reason codes — "
        + "timestamps, app usage durations, activity patterns, memory-store totals, "
        + "encryption state, exclusion counts, and a host label identifying this Mac "
        + "leave only when you ask ChatGPT to use an OpenBird tool, and cannot be "
        + "recalled by deleting local memory."
}
