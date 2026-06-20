"""Module entry point so the packaged .app can launch the CLI without a
console-script shebang (which would bake an absolute interpreter path).

The notarized .dmg bundles a relocatable Python and invokes
``python -m openbird`` from ``Contents/MacOS/openbird-cli``; baking no absolute
paths is what keeps the bundle relocatable to /Applications on a tester's Mac.
"""

from openbird.cli import app

if __name__ == "__main__":
    app()
