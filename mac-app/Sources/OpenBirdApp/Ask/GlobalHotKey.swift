import AppKit
import Carbon.HIToolbox

/// A process-wide global hotkey via Carbon `RegisterEventHotKey`. Carbon hotkeys
/// work on macOS 13+ and, unlike `NSEvent` global monitors, need **no** Accessibility
/// grant — important here, since requiring Accessibility just to summon the panel
/// would be a poor first-run experience. Failing to register (e.g. the combo is
/// already taken) returns nil; the caller degrades to the menu item.
final class GlobalHotKey {
    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?
    private let callback: () -> Void

    /// - Parameters:
    ///   - keyCode: a virtual key code (e.g. `kVK_Space`).
    ///   - modifiers: Carbon modifier mask (e.g. `optionKey`).
    ///   - callback: invoked on the main thread when the hotkey fires.
    init?(keyCode: UInt32, modifiers: UInt32, callback: @escaping () -> Void) {
        self.callback = callback

        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            { _, _, userData -> OSStatus in
                guard let userData else { return noErr }
                Unmanaged<GlobalHotKey>.fromOpaque(userData).takeUnretainedValue().callback()
                return noErr
            },
            1, &eventType, selfPtr, &eventHandler
        )
        guard installStatus == noErr else { return nil }

        // 'OBKY' signature keeps our hotkey id distinct from other apps'.
        let hotKeyID = EventHotKeyID(signature: OSType(0x4F424B59), id: 1)
        let registerStatus = RegisterEventHotKey(
            keyCode, modifiers, hotKeyID,
            GetApplicationEventTarget(), 0, &hotKeyRef
        )
        guard registerStatus == noErr, hotKeyRef != nil else {
            if let eventHandler { RemoveEventHandler(eventHandler) }
            eventHandler = nil
            return nil
        }
    }

    deinit {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
        if let eventHandler { RemoveEventHandler(eventHandler) }
    }
}
