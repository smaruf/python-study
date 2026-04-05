# MicroPythonOS — Zero to Expert

> **MicroPythonOS** is an Android-inspired, touch-UI operating environment built on
> [MicroPython](https://micropython.org/) and [LVGL](https://lvgl.io/).  
> It runs on ESP32-S3 touchscreen boards and can be tested locally on Linux.  
> It is **not** MicroPython's built-in `os` / `uos` module — it is a completely
> separate project.

---

## Table of Contents

| # | Topic | Link |
|---|-------|------|
| 1 | What is MicroPythonOS? | [this page §1](#1-what-is-micropythonos) |
| 2 | Prerequisites & required tools | [this page §2](#2-prerequisites--required-tools) |
| 3 | Install on Linux desktop | [docs/installation.md](docs/installation.md) |
| 4 | Flash firmware on ESP32 / ESP32-S3 | [docs/installation.md#esp32](docs/installation.md#flashing-onto-esp32--esp32-s3) |
| 5 | App development guide | [docs/app-development.md](docs/app-development.md) |
| 6 | Deploy apps with `mpremote` | [docs/deployment.md](docs/deployment.md) |
| 7 | Troubleshooting checklist | [docs/troubleshooting.md](docs/troubleshooting.md) |
| 8 | Links & further reading | [docs/resources.md](docs/resources.md) |

---

## 1. What is MicroPythonOS?

MicroPythonOS provides:

- **Touch UI** — built on LVGL; renders a home screen, app launcher, and settings
  similar to Android.
- **App framework** — apps are Python packages with a defined folder structure and a
  manifest.  Each app subclasses `mpos.Activity` and implements lifecycle hooks
  (`onCreate`, `onResume`, `onPause`, `onDestroy`).
- **App Store** — a curated catalogue of installable apps (no account required).
- **System services** — Wi-Fi management, OTA updates, display/touch abstraction,
  audio (where supported).
- **Desktop simulator** — the full OS can run on Linux for rapid app iteration before
  flashing to hardware.

### How it differs from MicroPython's `os` module

| | MicroPython `os` / `uos` | MicroPythonOS |
|---|---|---|
| What it is | Built-in module in MicroPython firmware | Separate project / firmware image |
| Purpose | POSIX-like file-system API | Android-style OS + UI framework |
| Typical import | `import os` or `import uos` | `import mpos` |
| Runs on | Any MicroPython port | ESP32-S3 boards + Linux desktop |
| UI | None | LVGL touch interface |

---

## 2. Prerequisites & Required Tools

### Hardware (optional — not needed for desktop testing)

| Item | Notes |
|------|-------|
| Waveshare ESP32-S3-Touch-LCD-2 (or similar ESP32-S3 board) | 240×320 or 320×480 IPS touch display recommended |
| USB-C cable | Data-capable; charging-only cables will not work |

### Software

| Tool | Version | Install |
|------|---------|---------|
| Python 3 | ≥ 3.9 | `sudo apt install python3` |
| `mpremote` | latest | `pip install mpremote` |
| `esptool.py` | latest | `pip install esptool` |
| MicroPythonOS Linux build | see [installation](docs/installation.md) | download from official releases |

> **Note:** `mpremote` is the recommended tool for copying files to the ESP32.
> It is part of the official MicroPython project.

---

## 3. Quick-Start

```bash
# 1. Clone the repo (you're already here if reading this in the repo)
cd projects/micropythonos

# 2. Read the installation guide
open docs/installation.md     # macOS
xdg-open docs/installation.md # Linux

# 3. Run the OS on your desktop
#    (see docs/installation.md for the full Linux setup steps)

# 4. Deploy the example countdown timer app
bash scripts/install_app.sh --app apps/com.smaruf.countdown_timer --target desktop
```

---

## 4. Project Layout

```
projects/micropythonos/
├── README.md                          ← you are here
├── docs/
│   ├── installation.md                ← Linux desktop + ESP32 flashing
│   ├── app-development.md             ← app structure, lifecycle, Hello World
│   ├── deployment.md                  ← mpremote, filesystem layout
│   ├── troubleshooting.md             ← common problems & fixes
│   └── resources.md                   ← links to upstream docs
├── apps/
│   └── com.smaruf.countdown_timer/    ← example app
│       ├── META-INF/
│       │   └── MANIFEST.JSON
│       ├── assets/
│       │   └── main.py                ← LVGL + mpos.Activity app code
│       └── res/
│           └── ICONS.md               ← icon generation instructions
└── scripts/
    ├── install_app.sh                 ← Bash helper (desktop + mpremote)
    └── install_app.py                 ← Cross-platform Python equivalent
```

---

## 5. Example App — Countdown Timer

The [`apps/com.smaruf.countdown_timer/`](apps/com.smaruf.countdown_timer/) folder
contains a fully-annotated example app that demonstrates:

- Correct `MANIFEST.JSON` structure
- Subclassing `mpos.Activity` with all lifecycle hooks
- Building a UI with LVGL widgets (label, button, arc)
- Handling touch events and state transitions
- Deploying to both desktop and ESP32 with the provided scripts

See [docs/app-development.md](docs/app-development.md) for a line-by-line walkthrough.

---

## 6. Authoritative References

- MicroPythonOS official docs: <https://docs.micropythonos.com/>
- Creating apps guide: <https://docs.micropythonos.com/app-development/creating-apps/>
- MicroPython ESP32 docs: <https://docs.micropython.org/en/latest/esp32/quickref.html>
- MicroPython downloads: <https://micropython.org/download/>
- DroneBot Workshop tutorial: <https://dronebotworkshop.com/micropythonos/>

---

*Back to repo root → [python-study](../../README.md)*
