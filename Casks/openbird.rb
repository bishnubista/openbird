cask "openbird" do
  version "0.2.0"
  sha256 "b8ef05ca9e553467ed0b1f1e92222aa15122bfe44db5da423e373a09d13f9b16"

  url "https://github.com/bishnubista/openbird/releases/download/beta-dmg-#{version}/OpenBird.dmg"
  name "OpenBird"
  desc "Local-first memory assistant with a native trust controller"
  homepage "https://github.com/bishnubista/openbird"

  # The notarized .dmg is the capture-capable artifact: a Developer ID signature
  # with a stable identity is what lets macOS grant (and persist) Screen Recording
  # and Accessibility (TCC). This is distinct from the `openbird` *formula*, which
  # installs the CLI only.
  depends_on macos: :ventura # minimum macOS 13 (app deployment target)

  app "OpenBird.app"

  uninstall quit: "ai.openbird.OpenBird"

  # `zap` mirrors what the in-app/CLI uninstaller (`openbird uninstall --purge-data`)
  # removes from disk, so `brew uninstall --cask --zap openbird` no longer leaves the
  # routines daemon loaded or on-device state behind. Each directive maps to a step in
  # openbird/uninstall.py:
  #   - launchctl: boots out the routines launchd job  (remove_routines_job /
  #     _boot_out_routines; label = routines.launchd.AGENT_LABEL "ai.openbird.routines")
  #   - the *.plist trash: removes the routines LaunchAgent (agent_plist_path:
  #     ~/Library/LaunchAgents/ai.openbird.routines.plist)
  #   - ~/.openbird trash: removes the data dir + sidecars (data_dir_path default;
  #     covers remove_sidecars + _purge_data_dir)
  #
  # Deliberate limitations (do NOT expand beyond cask norms):
  #   - The macOS Keychain DB-encryption key is intentionally NOT zapped. The Python
  #     uninstaller deletes it only when no encrypted DB depends on it (_key_action);
  #     a cask zap has no such guard and could strand an encrypted DB on another
  #     install, so key handling stays with `openbird uninstall`.
  #   - Launch Services unregistration is left to macOS: `brew uninstall` removes the
  #     app bundle, and the Python path bundle-id-validates each entry before
  #     unregistering (unregister_launch_services) — a cask cannot replicate that guard.
  #   - Homebrew/Cellar/libexec paths are never touched here (managed by Homebrew).
  zap launchctl: "ai.openbird.routines",
      trash:     [
        "~/.openbird",
        "~/Library/LaunchAgents/ai.openbird.routines.plist",
      ]

  caveats <<~EOS
    OpenBird captures the text of your active window on-device. On first launch,
    Guided Setup walks you through granting Accessibility / Screen Recording
    permissions and provisioning a local Ollama model.

    Meeting audio transcription is NOT included in this notarized beta — that
    backend ships only when OpenBird is built from source with the `meetings`
    extra (see the project README). The download you install here does not
    capture meeting audio.

    `brew uninstall --cask openbird` removes the app; add `--zap` to also boot out
    the routines LaunchAgent and delete on-device memory (~/.openbird). The
    macOS Keychain DB-encryption key is left in place (run `openbird uninstall`
    for the full, key-aware cleanup).
  EOS
end
