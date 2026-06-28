import Foundation
import Security
import os

/// Resolves the OpenBird DB-encryption key from the login Keychain, owned by the
/// signed app itself. Because `OpenBird.app` has a stable Developer-ID designated
/// requirement and an `Info.plist` display name, the access prompt reads
/// "OpenBird" (not "python3.13") and "Always Allow" persists across launches. The
/// resolved key is injected into every CLI child via `OPENBIRD_DB_KEY`, so the
/// Python layer (`openbird/storage/crypto.py`) never touches `keyring` and never
/// raises its own prompt.
///
/// Design + Codex consensus: `docs/design/keychain-app-attribution.md`.
/// Privacy: only reason codes are logged — never the key material.
enum KeychainKeyProvider {
    /// Must match `crypto.py` `_KEYRING_SERVICE` / `_KEYRING_USER`.
    static let service = "openbird"
    static let account = "db-encryption-key"

    /// The 16-byte SQLite plaintext header. A DB file that does NOT begin with
    /// this is treated as encrypted (or otherwise unsafe to mint a new key for).
    private static let sqliteMagic = Array("SQLite format 3\u{0}".utf8)

    private static let log = Logger(subsystem: "ai.openbird.OpenBird", category: "keychain")

    /// Privacy-safe outcome reason codes (never includes the key).
    enum Outcome: String, Sendable {
        case loaded      // existing item read successfully
        case created     // generated + stored a fresh key (no/plaintext DB)
        case denied      // user denied/cancelled the read; left untouched
        case strandedDb  // encrypted-looking DB but no key item -> fail closed
        case error       // unexpected Keychain/error condition

        var wireCode: String {
            switch self {
            case .strandedDb: return "stranded_db"
            default: return rawValue
            }
        }
    }

    /// Resolve the DB key. Returns `nil` when no key can be *safely* provided.
    ///
    /// - Never deletes the existing item (Codex finding #1): a delete that
    ///   succeeded before a failed re-add would lose the only key and strand an
    ///   encrypted DB. We rely on the system "Always Allow" flow to add this app
    ///   to an existing item's ACL instead.
    /// - Generates a new key only when provably safe (Codex finding #2).
    static func resolveKey(dbPath: String) -> (key: String?, outcome: Outcome) {
        let (existing, status) = readKey()

        switch status {
        case errSecSuccess:
            if let key = existing, !key.isEmpty {
                log.info("db key resolved (\(Outcome.loaded.rawValue, privacy: .public))")
                return (key, .loaded)
            }
            log.error("db key item present but unreadable (\(Outcome.error.rawValue, privacy: .public))")
            return (nil, .error)

        case errSecItemNotFound:
            return generateIfSafe(dbPath: dbPath)

        case errSecAuthFailed, errSecUserCanceled, errSecInteractionNotAllowed:
            // Read denied/cancelled: do NOT generate — an existing encrypted DB
            // could be stranded by a fresh, non-matching key.
            log.error("db key read denied (\(Outcome.denied.rawValue, privacy: .public))")
            return (nil, .denied)

        default:
            log.error("db key lookup failed (\(Outcome.error.rawValue, privacy: .public)) status=\(status, privacy: .public)")
            return (nil, .error)
        }
    }

    /// Read the existing key item. Returns the decoded key (when the payload is
    /// present and valid UTF-8) plus the raw `SecItemCopyMatching` status so the
    /// caller can distinguish not-found / denied / error.
    private static func readKey() -> (key: String?, status: OSStatus) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess else { return (nil, status) }
        if let data = item as? Data, let key = String(data: data, encoding: .utf8) {
            return (key, status)
        }
        return (nil, status)  // success but payload missing / not UTF-8
    }

    /// Generate + store a fresh key, but only when no encrypted DB would be
    /// stranded. Otherwise fail closed.
    private static func generateIfSafe(dbPath: String) -> (key: String?, outcome: Outcome) {
        if looksEncrypted(dbPath: dbPath) {
            log.error("encrypted-looking DB with no key item; failing closed (\(Outcome.strandedDb.rawValue, privacy: .public))")
            return (nil, .strandedDb)
        }
        guard let key = randomHexKey() else {
            log.error("SecRandomCopyBytes failed (\(Outcome.error.rawValue, privacy: .public))")
            return (nil, .error)
        }
        let attrs: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecValueData as String: Data(key.utf8),
        ]
        let status = SecItemAdd(attrs as CFDictionary, nil)
        if status == errSecDuplicateItem {
            // A concurrent writer created the item between our not-found read and
            // this add. Re-read and use that key rather than failing bootstrap —
            // the item now exists and is the authoritative key.
            let (existing, readStatus) = readKey()
            if readStatus == errSecSuccess, let key = existing, !key.isEmpty {
                log.info("db key loaded after concurrent create (\(Outcome.loaded.rawValue, privacy: .public))")
                return (key, .loaded)
            }
            log.error("db key duplicate-add but re-read failed (\(Outcome.error.rawValue, privacy: .public)) status=\(readStatus, privacy: .public)")
            return (nil, .error)
        }
        guard status == errSecSuccess else {
            log.error("db key SecItemAdd failed (\(Outcome.error.rawValue, privacy: .public)) status=\(status, privacy: .public)")
            return (nil, .error)
        }
        log.info("db key generated (\(Outcome.created.rawValue, privacy: .public))")
        return (key, .created)
    }

    /// True iff the resolved DB file exists, is non-empty, and does NOT begin with
    /// the SQLite plaintext magic — i.e. it looks like a SQLCipher-encrypted DB
    /// whose key we must not overwrite. Absent or empty -> safe to create.
    static func looksEncrypted(dbPath: String) -> Bool {
        guard let handle = FileHandle(forReadingAtPath: dbPath) else {
            return false  // absent -> safe to mint a fresh key
        }
        defer { try? handle.close() }
        let head = handle.readData(ofLength: sqliteMagic.count)
        if head.isEmpty { return false }       // empty file -> safe
        return Array(head) != sqliteMagic      // non-empty & not plaintext -> treat as encrypted
    }

    /// 64-char lowercase hex of 32 random bytes — byte-for-byte the format Python
    /// produces via `secrets.token_hex(32)` and consumes verbatim as the SQLCipher
    /// key, so an existing encrypted DB stays decryptable.
    private static func randomHexKey() -> String? {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            return nil
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
}
