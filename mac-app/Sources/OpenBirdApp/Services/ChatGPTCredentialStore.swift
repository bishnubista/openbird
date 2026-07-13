import Foundation
import Security

struct ChatGPTTunnelCredential: Equatable, Sendable {
    let tunnelID: String
    let runtimeKey: String
}

/// Dedicated Keychain item for the ChatGPT tunnel. Neither value is written to
/// UserDefaults; the runtime key is injected only into the owned child process.
enum ChatGPTCredentialStore {
    private static let service = "openbird"
    private static let account = "chatgpt-secure-mcp-tunnel"

    static func load() -> ChatGPTTunnelCredential? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let values = try? JSONDecoder().decode([String: String].self, from: data),
              let tunnelID = values["tunnel_id"],
              let runtimeKey = values["runtime_key"],
              !tunnelID.isEmpty, !runtimeKey.isEmpty else { return nil }
        return ChatGPTTunnelCredential(tunnelID: tunnelID, runtimeKey: runtimeKey)
    }

    static func save(_ credential: ChatGPTTunnelCredential) -> Bool {
        guard let data = try? JSONEncoder().encode([
            "tunnel_id": credential.tunnelID,
            "runtime_key": credential.runtimeKey,
        ]) else { return false }
        let update = [kSecValueData as String: data]
        let status = SecItemUpdate(baseQuery as CFDictionary, update as CFDictionary)
        if status == errSecSuccess { return true }
        guard status == errSecItemNotFound else { return false }
        var attributes = baseQuery
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        attributes[kSecValueData as String] = data
        return SecItemAdd(attributes as CFDictionary, nil) == errSecSuccess
    }

    static func delete() -> Bool {
        let status = SecItemDelete(baseQuery as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    private static var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
    }
}
