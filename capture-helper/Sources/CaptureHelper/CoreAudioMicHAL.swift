// CoreAudioMicHAL — the production HAL seam behind MicMonitor (Phase C1).
//
// Thin, mechanism-only CoreAudio adapter: the aggregation/lifecycle POLICY
// lives in `CaptureHelperCore.MicMonitor` where it is unit-tested against a
// fake. Everything here is metadata: DeviceIsRunningSomewhere is public HAL
// state — no audio is read and no TCC prompt is triggered.
//
// All listener blocks are delivered on the MAIN dispatch queue so MicMonitor's
// state stays main-confined (the same confinement as the engine's scheduler
// input). Registration/removal must use the SAME property address, so each
// installed listener records the address it registered with.

import CaptureHelperCore
import CoreAudio
import Foundation

final class CoreAudioMicHAL: MicHAL {
    private static let systemObject = AudioObjectID(kAudioObjectSystemObject)

    /// Retains the device-list listener block for the helper's lifetime (the
    /// monitor never tears it down — stream mode exits by process death).
    private var deviceListBlock: AudioObjectPropertyListenerBlock?
    /// Per-device run-state listeners: the exact (address, block) pair used at
    /// registration, required verbatim for removal.
    private var runListeners:
        [MicDeviceID: (AudioObjectPropertyAddress, AudioObjectPropertyListenerBlock)] = [:]

    private static func address(
        _ selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: selector, mScope: scope,
            mElement: kAudioObjectPropertyElementMain)
    }

    func listDevices() -> [MicDeviceID] {
        var addr = Self.address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        guard
            AudioObjectGetPropertyDataSize(Self.systemObject, &addr, 0, nil, &size) == noErr,
            size > 0
        else { return [] }
        var ids = [AudioObjectID](
            repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard
            AudioObjectGetPropertyData(Self.systemObject, &addr, 0, nil, &size, &ids) == noErr
        else { return [] }
        return ids
    }

    func hasInputStreams(_ device: MicDeviceID) -> Bool {
        var addr = Self.address(
            kAudioDevicePropertyStreamConfiguration,
            scope: kAudioObjectPropertyScopeInput)
        var size: UInt32 = 0
        guard
            AudioObjectGetPropertyDataSize(device, &addr, 0, nil, &size) == noErr,
            size >= UInt32(MemoryLayout<AudioBufferList>.size)
        else { return false }
        let raw = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(device, &addr, 0, nil, &size, raw) == noErr
        else { return false }
        let list = raw.assumingMemoryBound(to: AudioBufferList.self)
        return UnsafeMutableAudioBufferListPointer(list).contains { $0.mNumberChannels > 0 }
    }

    private func runningSomewhere(
        _ device: MicDeviceID, scope: AudioObjectPropertyScope
    ) -> Bool? {
        var addr = Self.address(kAudioDevicePropertyDeviceIsRunningSomewhere, scope: scope)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &value) == noErr
        else { return nil }
        return value != 0
    }

    func isRunningInputScope(_ device: MicDeviceID) -> Bool? {
        runningSomewhere(device, scope: kAudioObjectPropertyScopeInput)
    }

    func isRunningGlobalScope(_ device: MicDeviceID) -> Bool {
        runningSomewhere(device, scope: kAudioObjectPropertyScopeGlobal) ?? false
    }

    func addDeviceListListener(_ handler: @escaping () -> Void) {
        let block: AudioObjectPropertyListenerBlock = { _, _ in handler() }
        var addr = Self.address(kAudioHardwarePropertyDevices)
        guard
            AudioObjectAddPropertyListenerBlock(
                Self.systemObject, &addr, DispatchQueue.main, block) == noErr
        else {
            // Degraded but honest: startup enumeration still works; device
            // churn (AirPods) just won't be tracked until restart.
            diag("capture: mic_device_list_listener_failed")
            return
        }
        deviceListBlock = block
    }

    func addRunStateListener(_ device: MicDeviceID, handler: @escaping () -> Void) {
        let block: AudioObjectPropertyListenerBlock = { _, _ in handler() }
        // Input scope first (matches the query); fall back to the global-scope
        // address so a scope-quirky HAL still delivers change notifications.
        // The monitor's per-read input-scope query keeps output-only playback
        // from reading as mic-hot even when notifications arrive globally.
        var addr = Self.address(
            kAudioDevicePropertyDeviceIsRunningSomewhere,
            scope: kAudioObjectPropertyScopeInput)
        if AudioObjectAddPropertyListenerBlock(device, &addr, DispatchQueue.main, block)
            != noErr
        {
            addr = Self.address(kAudioDevicePropertyDeviceIsRunningSomewhere)
            guard
                AudioObjectAddPropertyListenerBlock(
                    device, &addr, DispatchQueue.main, block) == noErr
            else {
                diag("capture: mic_run_listener_failed")
                return
            }
        }
        runListeners[device] = (addr, block)
    }

    func removeRunStateListener(_ device: MicDeviceID) {
        guard var entry = runListeners.removeValue(forKey: device) else { return }
        AudioObjectRemovePropertyListenerBlock(device, &entry.0, DispatchQueue.main, entry.1)
    }
}
