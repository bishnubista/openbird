class Openbird < Formula
  desc "Local-first macOS memory assistant with a native trust controller"
  homepage "https://github.com/bishnubista/openbird"
  # OpenBird is a PRIVATE repo, so the browser release-download URL 404s without
  # auth. Fetch the tarball from the GitHub API asset endpoint instead, which
  # honors a token: set HOMEBREW_GITHUB_API_TOKEN (a PAT with `repo` scope, e.g.
  # `export HOMEBREW_GITHUB_API_TOKEN=$(gh auth token)`) before installing. On
  # GitHub's redirect to pre-signed storage, Homebrew drops the Authorization
  # header automatically (different host), so the token never leaks downstream.
  url "https://api.github.com/repos/bishnubista/openbird/releases/assets/451977727",
      headers: [
        "Accept: application/octet-stream",
        "Authorization: token #{ENV.fetch("HOMEBREW_GITHUB_API_TOKEN", "")}",
      ]
  version "0.1.4"
  sha256 "57a5f5f6e62bc110d00655adbb1c57877b259b2deae14b046e346ec680e12e27"

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

    # Ad-hoc sign the assembled bundle with stable identifiers so macOS TCC grants
    # (Accessibility / Screen Recording / Microphone) persist across launches.
    # Ad-hoc needs no keychain, so it works inside the $HOME-redirected
    # `brew install` environment (a self-signed keychain cert cannot — Homebrew
    # isolates HOME, and it would also trigger GUI keychain prompts). The script
    # ships in the same archive as this formula (git ls-files); the existence
    # guard is purely defensive against the script ever being absent.
    system "./script/sign_local.sh", app if File.exist?("script/sign_local.sh")

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

      OpenBird.app is ad-hoc signed at install time with stable identifiers, so
      macOS TCC grants (Accessibility / Screen Recording / Microphone) persist
      across launches — you grant them once. A `brew upgrade` rebuilds the app and
      changes its code hash, so you re-grant permissions once after an upgrade.
      The app is NOT notarized, so on first launch macOS Gatekeeper may warn —
      right-click the app and choose Open, or approve it in System Settings >
      Privacy & Security.

      To enable screen/audio capture, open the app and follow Guided Setup: it
      walks you through Ollama, first-launch local model download, and granting
      Accessibility / Screen Recording / Microphone permissions. The default
      Ollama route downloads the RAM-tiered qwen3 model (qwen3:4b on ~16 GB Macs,
      qwen3:8b on 24/32 GB) + embeddinggemma (needs Ollama >= 0.11.10; roughly
      3-6 GB) into
      Ollama-managed storage. macOS requires you to toggle permissions yourself
      in System Settings; the app deep-links you to the right pane and re-checks.
      The CLI memory features (openbird ingest / chat / routine) work immediately
      without any permissions.
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
