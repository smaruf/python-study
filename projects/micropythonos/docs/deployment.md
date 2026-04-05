# Deployment Guide

This guide covers how to deploy MicroPythonOS apps to both the Linux desktop
simulator and a real ESP32 / ESP32-S3 device using `mpremote`.

---

## Table of Contents

- [Filesystem layout](#filesystem-layout)
- [Deploy to Linux desktop](#deploy-to-linux-desktop)
- [Deploy to ESP32 with mpremote](#deploy-to-esp32-with-mpremote)
  - [Install mpremote](#install-mpremote)
  - [Connect and explore the device filesystem](#connect-and-explore-the-device-filesystem)
  - [Copy an app to the device](#copy-an-app-to-the-device)
  - [Verify the install](#verify-the-install)
- [Remove an app](#remove-an-app)
- [OTA (Over-the-Air) updates](#ota-over-the-air-updates)
- [Helper script reference](#helper-script-reference)

---

## Filesystem layout

MicroPythonOS uses **LittleFS** on the ESP32 internal flash.
The top-level layout on device looks like this:

```
/
├── apps/                      ← user-installed apps
│   ├── com.smaruf.countdown_timer/
│   │   ├── META-INF/MANIFEST.JSON
│   │   ├── assets/main.py
│   │   └── res/icon-48.png
│   └── com.example.another_app/
│       └── ...
├── system/                    ← OS system files (do not modify)
├── settings/                  ← persisted user settings
└── ...
```

The same layout is mirrored on desktop at:

```
~/.micropythonos/internal_filesystem/
```

---

## Deploy to Linux desktop

```bash
# Copy the app bundle
cp -r projects/micropythonos/apps/com.smaruf.countdown_timer \
      ~/.micropythonos/internal_filesystem/apps/

# Restart the OS to pick up the new app
```

Or use the provided helper:

```bash
bash projects/micropythonos/scripts/install_app.sh \
  --app projects/micropythonos/apps/com.smaruf.countdown_timer \
  --target desktop
```

---

## Deploy to ESP32 with mpremote

### Install mpremote

```bash
pip install mpremote
```

Verify the install:

```bash
mpremote --version
```

### Connect and explore the device filesystem

```bash
PORT=/dev/ttyUSB0   # adjust to your port

# List the root filesystem
mpremote connect $PORT ls :

# List the apps directory
mpremote connect $PORT ls :/apps/
```

### Copy an app to the device

`mpremote cp -r` copies an entire directory tree recursively.
The destination must begin with `:` (the device).

```bash
PORT=/dev/ttyUSB0
APP_LOCAL=projects/micropythonos/apps/com.smaruf.countdown_timer

# Create the apps directory if it doesn't exist
mpremote connect $PORT mkdir :/apps/

# Copy the app bundle
mpremote connect $PORT cp -r $APP_LOCAL :/ apps/com.smaruf.countdown_timer
```

> **Tip:** `mpremote` copies each file individually.  A bundle with many small files
> takes longer than a bundle with fewer larger files.  Keep `assets/` lean.

After copying, soft-reset the device to reload the app list:

```bash
mpremote connect $PORT reset
```

Or from the REPL:

```bash
mpremote connect $PORT repl
# then press Ctrl+D for soft reset, Ctrl+X to exit
```

### Verify the install

```bash
mpremote connect $PORT ls :/apps/com.smaruf.countdown_timer/
# Expected output:
#   META-INF/
#   assets/
#   res/
```

---

## Remove an app

```bash
PORT=/dev/ttyUSB0

# Remove recursively using mpremote exec
mpremote connect $PORT exec "import os; \
  def rmtree(p): \
    for f in os.listdir(p): \
      fp = p+'/'+f; \
      (rmtree(fp) if (os.stat(fp)[0] & 0x4000) else os.remove(fp)); \
    os.rmdir(p); \
  rmtree('/apps/com.smaruf.countdown_timer')"
```

---

## OTA (Over-the-Air) updates

MicroPythonOS includes a built-in **Settings → System Update** screen that downloads
and installs OS updates over Wi-Fi.  For individual app updates, re-run the
`mpremote cp` command — it overwrites existing files.

---

## Helper script reference

| Script | Purpose |
|--------|---------|
| `scripts/install_app.sh` | Bash: copy to desktop or device via mpremote |
| `scripts/install_app.py` | Python: cross-platform equivalent |

### `install_app.sh` flags

```
Usage: install_app.sh [OPTIONS]

Options:
  --app <path>        Path to the app bundle directory (required)
  --target <target>   Either 'desktop' or a serial port, e.g. /dev/ttyUSB0
  --help              Show this help message
```

### `install_app.py` flags

```
Usage: python install_app.py [OPTIONS]

Options:
  --app PATH          Path to the app bundle directory (required)
  --target TARGET     'desktop' or a serial port path (required)
  --help              Show this help message
```

---

*← [App Development](app-development.md) | [Troubleshooting →](troubleshooting.md)*
