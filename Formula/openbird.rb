class Openbird < Formula
  desc "Local-first macOS memory assistant with a native trust controller"
  homepage "https://github.com/bishnubista/openbird"
  url "https://github.com/bishnubista/openbird/releases/download/v0.1.0/openbird-0.1.0.tar.gz"
  sha256 "f42bbdc702ebdf40f18205f3e9c0d3223cf11039b986bdbc4f5cb6b92774e6c3"

  depends_on "uv" => :build
  depends_on :macos
  depends_on "python@3.13"

  # OpenBird installs a source-built venv plus local Swift products.
  # Preserve Python wheel @rpath IDs so Homebrew does not rewrite bundled
  # extension modules with longer install names than their headers can hold.
  preserve_rpath

  def install
    odie "OpenBird requires Apple's Swift toolchain. Install Xcode Command Line Tools first." unless which("swift")

    python = Formula["python@3.13"].opt_bin/"python3.13"
    venv = libexec/"venv"

    system "uv", "venv", "--python", python, venv
    system "uv", "pip", "install", "--python", venv/"bin/python", ".[encryption]"

    ENV["OPENBIRD_SWIFTPM_DISABLE_SANDBOX"] = "1"
    # Build the bundle but defer signing: we rewrite openbird-cli below, and
    # signing must happen AFTER that rewrite or the app's seal would be invalid.
    ENV["OPENBIRD_SKIP_SIGN"] = "1"
    system "./script/build_and_run.sh", "--no-launch"
    app = buildpath/"dist/OpenBird.app"
    app_macos = app/"Contents/MacOS"

    rm app_macos/"openbird-cli"
    (app_macos/"openbird-cli").write <<~SH
      #!/usr/bin/env bash
      set -euo pipefail
      BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
      export OPENBIRD_CAPTURE_HELPER="$BIN_DIR/capture-helper"
      export OPENBIRD_AUDIO_HELPER="$BIN_DIR/audio-helper"
      exec "#{bin}/openbird" "$@"
    SH
    chmod 0755, app_macos/"openbird-cli"

    # Sign the assembled bundle with a stable, self-signed local identity so macOS
    # TCC grants (Accessibility / Screen Recording / Microphone) persist across
    # rebuilds. Fail-soft: sign_local.sh degrades to ad-hoc signing if the
    # self-signed identity cannot be created, so install never breaks.
    system "./script/sign_local.sh", app

    libexec.install app

    (bin/"openbird").write <<~SH
      #!/usr/bin/env bash
      set -euo pipefail
      exec "#{venv}/bin/openbird" "$@"
    SH
    chmod 0755, bin/"openbird"

    (bin/"openbird-app").write <<~SH
      #!/usr/bin/env bash
      set -euo pipefail
      exec /usr/bin/open -n "#{libexec}/OpenBird.app"
    SH
    chmod 0755, bin/"openbird-app"
  end

  def caveats
    <<~EOS
      OpenBird.app was installed to:
        #{libexec}/OpenBird.app

      Launch it with:
        openbird-app

      This install built from source and pulled Python dependencies from PyPI
      (uv pip install). It is not vendored/offline and is not pinned for
      reproducible offline installs; network access is required at install time.

      OpenBird.app is signed at install time with a STABLE, SELF-SIGNED local
      identity (stored in the openbird-codesign keychain). This gives the bundle a
      consistent code-signing identity so macOS TCC grants persist across
      rebuilds. The app and helpers are NOT notarized, so on first launch macOS
      Gatekeeper may warn — right-click the app and choose Open, or approve it in
      System Settings > Privacy & Security.

      To enable screen/audio capture, open the app and follow Guided Setup: it
      walks you through Ollama, models, and granting Accessibility / Screen
      Recording / Microphone permissions. macOS requires you to toggle those
      permissions yourself in System Settings; the app deep-links you to the right
      pane and re-checks. The CLI memory features (openbird ingest / chat /
      routine) work immediately without any permissions.
    EOS
  end

  test do
    system bin/"openbird", "--help"
    assert_path_exists libexec/"OpenBird.app/Contents/MacOS/OpenBird"
    assert_path_exists libexec/"OpenBird.app/Contents/MacOS/capture-helper"
    assert_path_exists libexec/"OpenBird.app/Contents/MacOS/audio-helper"
    assert_predicate bin/"openbird-app", :executable?
  end
end
