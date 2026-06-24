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

  zap trash: [
    "~/.openbird",
    "~/Library/LaunchAgents/ai.openbird.routines.plist",
  ]

  caveats <<~EOS
    OpenBird captures the text of your active window and meeting audio on-device.
    On first launch, Guided Setup walks you through granting Accessibility /
    Screen Recording / Microphone permissions and provisioning a local Ollama
    model. `brew uninstall --cask openbird` removes the app; add `--zap` to also
    delete on-device memory (~/.openbird) and the routines LaunchAgent.
  EOS
end
