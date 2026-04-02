# Installation Guide

This guide covers two install paths:

1. **Linux desktop** — run MicroPythonOS inside a window on your PC for fast app development.
2. **ESP32 / ESP32-S3** — flash the firmware to real hardware.

---

## Table of Contents

- [Linux Desktop Install](#linux-desktop-install)
  - [System requirements](#system-requirements)
  - [Download the Linux build](#download-the-linux-build)
  - [Run the OS](#run-the-os)
  - [Filesystem location on desktop](#filesystem-location-on-desktop)
- [Flashing onto ESP32 / ESP32-S3](#flashing-onto-esp32--esp32-s3)
  - [Requirements](#requirements)
  - [Step 1 — Download firmware](#step-1--download-firmware)
  - [Step 2 — Identify the serial port](#step-2--identify-the-serial-port)
  - [Step 3 — Erase and flash](#step-3--erase-and-flash)
  - [Step 4 — Verify](#step-4--verify)
- [Web Installer (beginner option)](#web-installer-beginner-option)

---

## Linux Desktop Install

### System requirements

| Requirement | Notes |
|-------------|-------|
| OS | Ubuntu 20.04 / 22.04 / 24.04 LTS (or Debian-based equivalent) |
| Python | ≥ 3.9 |
| SDL2 library | `sudo apt install libsdl2-2.0-0` |
| Display | Any — the OS renders inside an SDL2 window |

Install SDL2 if you don't have it:

```bash
sudo apt update && sudo apt install -y libsdl2-2.0-0 libsdl2-dev
```

### Download the Linux build

1. Go to the MicroPythonOS releases page:
   <https://docs.micropythonos.com/os-development/installing-on-linux/>
2. Download the latest `micropythonos-linux-*.zip` (or `*.tar.gz`) archive.
3. Extract it:

```bash
mkdir -p ~/micropythonos && cd ~/micropythonos
# replace the filename with the version you downloaded
unzip micropythonos-linux-*.zip
```

### Run the OS

```bash
cd ~/micropythonos
./micropythonos          # or: python3 micropythonos.py  (depends on release)
```

An SDL2 window opens simulating the ESP32-S3 display (320×480 by default).
You can click with your mouse as if using a touchscreen.

### Filesystem location on desktop

When running on Linux the virtual device filesystem lives at:

```
~/.micropythonos/internal_filesystem/
```

Key sub-directories:

| Path | Purpose |
|------|---------|
| `internal_filesystem/apps/` | Installed user apps |
| `internal_filesystem/system/` | OS system files (do not edit) |
| `internal_filesystem/settings/` | Persisted settings |

To install an app on desktop, copy the app bundle into the `apps/` folder:

```bash
cp -r projects/micropythonos/apps/com.smaruf.countdown_timer \
      ~/.micropythonos/internal_filesystem/apps/
```

Then restart the OS (close and re-open) and the app appears in the launcher.

---

## Flashing onto ESP32 / ESP32-S3

### Requirements

- `esptool.py` — `pip install esptool`
- `mpremote` — `pip install mpremote`
- A supported ESP32-S3 board (e.g., Waveshare ESP32-S3-Touch-LCD-2)
- USB-C data cable

### Step 1 — Download firmware

1. Visit <https://docs.micropythonos.com/os-development/installing-on-esp32/>
2. Download the `.bin` firmware file that matches your board.

> **Important:** MicroPythonOS ships its own firmware image that already includes
> LVGL, touch drivers, and the mpos framework.  Do **not** flash a plain MicroPython
> firmware — you would lose the OS layer.

### Step 2 — Identify the serial port

Connect the board via USB then:

```bash
# Linux
ls /dev/ttyUSB* /dev/ttyACM*

# Or use dmesg
dmesg | grep -i tty | tail -5
```

The port is usually `/dev/ttyUSB0` or `/dev/ttyACM0`.

If you get a permission error:

```bash
sudo usermod -aG dialout $USER
# Log out and back in for the change to take effect
```

### Step 3 — Erase and flash

```bash
PORT=/dev/ttyUSB0        # adjust to your port
FIRMWARE=micropythonos-esp32s3-*.bin   # adjust to downloaded filename

# Erase flash first (recommended for a clean install)
esptool.py --chip esp32s3 --port $PORT erase_flash

# Flash the firmware
esptool.py --chip esp32s3 --port $PORT --baud 460800 write_flash 0x0 $FIRMWARE
```

The flash process takes roughly 30–60 seconds.  The board resets automatically.

### Step 4 — Verify

After flashing, the MicroPythonOS home screen should appear on the display.
You can also check the REPL:

```bash
mpremote connect $PORT repl
# Press Ctrl+C to interrupt running code; you should see a MicroPython prompt
# Press Ctrl+D to soft-reset
# Press Ctrl+X to exit mpremote
```

---

## Web Installer (beginner option)

MicroPythonOS provides a browser-based installer (Chrome / Edge with WebSerial):

1. Open <https://docs.micropythonos.com/os-development/installing-on-esp32/>
2. Click **Install via Web**.
3. Select your board type.
4. Follow the on-screen instructions.

This is the easiest approach for first-time users.

---

*← Back to [README](../README.md) | Next: [App Development →](app-development.md)*
