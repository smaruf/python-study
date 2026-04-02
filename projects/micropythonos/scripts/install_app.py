#!/usr/bin/env python3
"""
install_app.py
==============
Cross-platform helper to install a MicroPythonOS app bundle to either:
  - the Linux/macOS desktop simulator's virtual filesystem, or
  - a connected ESP32 device via mpremote.

Usage:
    python install_app.py --app <path_to_app_bundle> --target <desktop|PORT>

Examples:
    # Install to desktop simulator
    python install_app.py --app ../apps/com.smaruf.countdown_timer --target desktop

    # Install to ESP32 on Linux
    python install_app.py --app ../apps/com.smaruf.countdown_timer --target /dev/ttyUSB0

    # Install to ESP32 on Windows
    python install_app.py --app ..\\apps\\com.smaruf.countdown_timer --target COM3

    # Install to ESP32 on macOS
    python install_app.py --app ../apps/com.smaruf.countdown_timer --target /dev/cu.usbserial-0001

Requirements:
    - Python 3.9+
    - mpremote (pip install mpremote) — only needed for device installs
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def desktop_fs_path() -> Path:
    """Return the path to the desktop simulator's internal filesystem."""
    system = platform.system()
    if system in ("Linux", "Darwin"):
        return Path.home() / ".micropythonos" / "internal_filesystem"
    elif system == "Windows":
        return Path.home() / "AppData" / "Roaming" / "micropythonos" / "internal_filesystem"
    else:
        die(f"Unsupported platform: {system}")


def validate_app(app_path: Path) -> str:
    """Validate the app bundle and return the bundle ID."""
    if not app_path.is_dir():
        die(f"App path does not exist or is not a directory: {app_path}")

    manifest_path = app_path / "META-INF" / "MANIFEST.JSON"
    if not manifest_path.exists():
        die(f"MANIFEST.JSON not found at {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"MANIFEST.JSON is not valid JSON: {exc}")

    required_fields = ("package", "name", "version", "description", "author", "entry")
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        die(f"MANIFEST.JSON is missing required fields: {missing}")

    bundle_id: str = manifest["package"]
    if bundle_id != app_path.name:
        print(
            f"WARNING: bundle ID '{bundle_id}' differs from folder name '{app_path.name}'. "
            "The folder name is used as the install target.",
            file=sys.stderr,
        )

    return bundle_id


# ---------------------------------------------------------------------------
# Install to desktop
# ---------------------------------------------------------------------------

def install_desktop(app_path: Path) -> None:
    fs = desktop_fs_path()

    if not fs.exists():
        die(
            f"Desktop filesystem not found at {fs}. "
            "Run MicroPythonOS on this machine at least once to create it."
        )

    apps_dir = fs / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)

    dest = apps_dir / app_path.name
    print(f"Installing '{app_path.name}' to desktop simulator...")
    print(f"  Source : {app_path}")
    print(f"  Dest   : {dest}")

    if dest.exists():
        print(f"  Removing existing install at {dest}")
        shutil.rmtree(dest)

    shutil.copytree(app_path, dest)
    print("  Done.  Restart MicroPythonOS desktop to see the app.")


# ---------------------------------------------------------------------------
# Install to device via mpremote
# ---------------------------------------------------------------------------

def run_mpremote(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run an mpremote command and return the result."""
    cmd = ["mpremote"] + list(args)
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def install_device(app_path: Path, port: str) -> None:
    if shutil.which("mpremote") is None:
        die("'mpremote' is not installed. Run: pip install mpremote")

    bundle_name = app_path.name
    print(f"Installing '{bundle_name}' to device on {port} ...")
    print(f"  Source : {app_path}")
    print(f"  Dest   : :/apps/{bundle_name}")

    # Ensure /apps exists on device
    ensure_apps_dir = (
        "import os\n"
        "try:\n"
        "    os.stat('/apps')\n"
        "except OSError:\n"
        "    os.mkdir('/apps')\n"
        "    print('Created /apps')\n"
    )
    result = run_mpremote("connect", port, "exec", ensure_apps_dir)
    if result.stdout:
        print(" ", result.stdout.strip())

    # Remove existing install
    rmtree_code = (
        "import os\n"
        "\n"
        "def _rmtree(path):\n"
        "    try:\n"
        "        entries = os.listdir(path)\n"
        "    except OSError:\n"
        "        return\n"
        "    for entry in entries:\n"
        "        full = path + '/' + entry\n"
        "        try:\n"
        "            if os.stat(full)[0] & 0x4000:\n"
        "                _rmtree(full)\n"
        "            else:\n"
        "                os.remove(full)\n"
        "        except OSError as exc:\n"
        "            print('Warning removing', full, ':', exc)\n"
        "    os.rmdir(path)\n"
        "\n"
        f"dest = '/apps/{bundle_name}'\n"
        "try:\n"
        "    os.stat(dest)\n"
        "    print('Removing existing install:', dest)\n"
        "    _rmtree(dest)\n"
        "except OSError:\n"
        "    pass\n"
    )
    result = run_mpremote("connect", port, "exec", rmtree_code)
    if result.stdout:
        print(" ", result.stdout.strip())

    # Copy app bundle
    run_mpremote("connect", port, "cp", "-r", str(app_path), f":/apps/{bundle_name}")

    # Soft-reset device
    print("  Soft-resetting device...")
    run_mpremote("connect", port, "reset", check=False)

    print("  Done.  Your app should appear in the MicroPythonOS launcher.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--app",
        required=True,
        metavar="PATH",
        help="Path to the app bundle directory.",
    )
    parser.add_argument(
        "--target",
        required=True,
        metavar="TARGET",
        help="'desktop' or a serial port path (e.g. /dev/ttyUSB0, COM3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_path = Path(args.app).resolve()
    target = args.target

    validate_app(app_path)

    if target == "desktop":
        install_desktop(app_path)
    else:
        install_device(app_path, target)


if __name__ == "__main__":
    main()
