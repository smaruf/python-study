#!/usr/bin/env bash
# =============================================================================
# install_app.sh
# =============================================================================
# Copy a MicroPythonOS app bundle to either:
#   - the Linux desktop simulator's virtual filesystem, or
#   - a connected ESP32 device via mpremote.
#
# Usage:
#   install_app.sh --app <path_to_app_bundle> --target <desktop|/dev/ttyUSBx>
#
# Examples:
#   # Install to desktop simulator
#   ./install_app.sh --app ../apps/com.smaruf.countdown_timer --target desktop
#
#   # Install to ESP32 on /dev/ttyUSB0
#   ./install_app.sh --app ../apps/com.smaruf.countdown_timer --target /dev/ttyUSB0
#
# Requirements:
#   - bash ≥ 4
#   - mpremote (pip install mpremote) — only needed for device installs
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults & helpers
# ---------------------------------------------------------------------------

DESKTOP_FS="${HOME}/.micropythonos/internal_filesystem"
APP_PATH=""
TARGET=""

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
    exit 0
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)    APP_PATH="$2"; shift 2 ;;
        --target) TARGET="$2";   shift 2 ;;
        --help|-h) usage ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -z "$APP_PATH" ]] && die "--app is required"
[[ -z "$TARGET" ]]   && die "--target is required (use 'desktop' or a serial port)"
[[ -d "$APP_PATH" ]] || die "App path does not exist or is not a directory: $APP_PATH"

# Derive bundle name from the directory name
BUNDLE_NAME="$(basename "$APP_PATH")"

# Validate that a MANIFEST.JSON is present
MANIFEST="$APP_PATH/META-INF/MANIFEST.JSON"
[[ -f "$MANIFEST" ]] || die "MANIFEST.JSON not found at $MANIFEST"

# ---------------------------------------------------------------------------
# Install to desktop simulator
# ---------------------------------------------------------------------------

install_desktop() {
    local dest="$DESKTOP_FS/apps/$BUNDLE_NAME"

    echo "Installing '$BUNDLE_NAME' to desktop simulator..."
    echo "  Source : $APP_PATH"
    echo "  Dest   : $dest"

    if [[ ! -d "$DESKTOP_FS" ]]; then
        die "Desktop filesystem not found at $DESKTOP_FS. " \
            "Have you run MicroPythonOS on this machine at least once?"
    fi

    mkdir -p "$DESKTOP_FS/apps"

    if [[ -d "$dest" ]]; then
        echo "  Removing existing install at $dest"
        rm -rf "$dest"
    fi

    cp -r "$APP_PATH" "$dest"
    echo "  Done.  Restart MicroPythonOS desktop to see the app."
}

# ---------------------------------------------------------------------------
# Install to ESP32 device via mpremote
# ---------------------------------------------------------------------------

install_device() {
    local port="$TARGET"

    command -v mpremote >/dev/null 2>&1 \
        || die "'mpremote' is not installed. Run: pip install mpremote"

    echo "Installing '$BUNDLE_NAME' to device on $port ..."
    echo "  Source : $APP_PATH"
    echo "  Dest   : :/apps/$BUNDLE_NAME"

    # Ensure /apps exists on device
    mpremote connect "$port" exec "
import os
try:
    os.stat('/apps')
except OSError:
    os.mkdir('/apps')
    print('Created /apps')
"

    # Remove existing install to ensure a clean copy
    mpremote connect "$port" exec "
import os

def _rmtree(path):
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for entry in entries:
        full = path + '/' + entry
        try:
            if os.stat(full)[0] & 0x4000:
                _rmtree(full)
            else:
                os.remove(full)
        except OSError as e:
            print('Warning removing', full, ':', e)
    os.rmdir(path)

dest = '/apps/$BUNDLE_NAME'
try:
    os.stat(dest)
    print('Removing existing install:', dest)
    _rmtree(dest)
except OSError:
    pass
"

    # Copy app bundle to device
    mpremote connect "$port" cp -r "$APP_PATH" ":/apps/$BUNDLE_NAME"

    echo "  Soft-resetting device..."
    mpremote connect "$port" reset || true

    echo "  Done.  Your app should appear in the MicroPythonOS launcher."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if [[ "$TARGET" == "desktop" ]]; then
    install_desktop
else
    install_device
fi
