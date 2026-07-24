# frozen_string_literal: true

cask "openbird" do
  version "0.14.0"
  sha256 "d7e40f12d423d83b7451f47e625f0ac0d3f89d5ea8a8cabc7d27aec6e1166f18"

  url "https://github.com/bishnubista/openbird/releases/download/beta-dmg-#{version}/OpenBird.dmg"
  name "OpenBird"
  desc "Local-first memory assistant with a native trust controller"
  homepage "https://github.com/bishnubista/openbird"

  # The notarized .dmg is the capture-capable artifact: a Developer ID signature
  # with a stable identity is what lets macOS grant (and persist) Screen Recording
  # and Accessibility (TCC). This is distinct from the `openbird` *formula*, which
  # installs the CLI only.
  depends_on arch:  :arm64
  depends_on macos: :ventura # minimum macOS 13 (app deployment target)

  app "OpenBird.app"
  # Symlink the app's bundled CLI onto PATH. `openbird` ships as BOTH this cask and a
  # same-token formula; Homebrew intentionally skips auto-linking the formula's
  # bin/openbird while the cask is installed ("cask is installed, skipping link"),
  # silently leaving `openbird` off PATH until a manual `brew link`. This stanza makes
  # the cask itself own the PATH entry. Safe only because the bundled openbird-cli
  # wrapper resolves $0 through its symlink chain (PR #161, first shipped in the
  # 0.6.1 dmg) — earlier dmgs would resolve the interpreter next to the symlink and break.
  binary "#{appdir}/OpenBird.app/Contents/MacOS/openbird-cli", target: "openbird"

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

    This notarized app requires Apple Silicon and supports manual meeting recording with
    ScreenCaptureKit system audio plus microphone audio. Recording starts only
    after you click Start and acknowledge participant consent. First use asks
    before downloading the approximately 2.51 GB local Parakeet model; raw audio
    is not persisted. The Homebrew formula remains a portable CLI and does not
    include this meeting backend.

    `brew uninstall --cask openbird` removes the app; add `--zap` to also boot out
    the routines LaunchAgent and delete on-device memory (~/.openbird). The
    macOS Keychain DB-encryption key is left in place (run `openbird uninstall`
    for the full, key-aware cleanup).
  EOS
end
