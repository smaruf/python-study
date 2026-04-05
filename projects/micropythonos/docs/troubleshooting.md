# Troubleshooting Checklist

Work through the relevant section below before opening an issue or asking for help.

---

## Table of Contents

- [Desktop (Linux) issues](#desktop-linux-issues)
- [ESP32 flashing issues](#esp32-flashing-issues)
- [mpremote / serial issues](#mpremote--serial-issues)
- [App not appearing in launcher](#app-not-appearing-in-launcher)
- [App crashes on launch](#app-crashes-on-launch)
- [LVGL / display issues](#lvgl--display-issues)
- [Getting more diagnostic information](#getting-more-diagnostic-information)

---

## Desktop (Linux) issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `error while loading shared libraries: libSDL2` | SDL2 not installed | `sudo apt install libsdl2-2.0-0` |
| Blank window / no UI | Wrong build for your architecture | Download the correct Linux build (x86_64 vs ARM) |
| `Permission denied` when running binary | Binary not executable | `chmod +x ./micropythonos` |
| OS window very small | HiDPI scaling issue | Try setting `SDL_VIDEODRIVER=x11` before running: `SDL_VIDEODRIVER=x11 ./micropythonos` |
| App changes not reflected after copy | OS is still running with old state | Close and relaunch the desktop OS after copying apps |

---

## ESP32 flashing issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `esptool.py: error: Could not open /dev/ttyUSB0` | Wrong port or device not detected | Re-plug the cable; verify port with `dmesg | grep tty` |
| Permission denied on `/dev/ttyUSB*` | User not in `dialout` group | `sudo usermod -aG dialout $USER` then log out and back in |
| Flash fails mid-way | Unreliable USB cable or hub | Use a direct, data-capable USB-C cable; avoid hubs |
| Board not entering flash mode | Boot button not held | Hold `BOOT` (GPIO0) button while connecting USB; release after `esptool` connects |
| After flashing, display shows garbage | Wrong firmware variant | Download the firmware matching your exact board model |
| After flashing, nothing on display | Firmware for wrong chip | Confirm chip: `esptool.py --port $PORT chip_id` |

---

## mpremote / serial issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `mpremote: failed to connect` | Port busy or wrong | Another process (Thonny, screen) has the port open — close it first |
| `OSError: [Errno 13] EACCES` | Permission denied on port | Add yourself to `dialout` group (see above) |
| Files copy very slowly | Normal for LittleFS over serial | Be patient; or increase baud: `mpremote connect $PORT` uses 115200 by default |
| `mpremote cp` reports success but file is empty | Disk full on device | Check free space: `mpremote connect $PORT exec "import os; print(os.statvfs('/'))"` |

---

## App not appearing in launcher

Go through this checklist in order:

- [ ] The app folder is inside `/apps/` (not a sub-folder of `system/` etc.).
- [ ] The folder name matches the `package` field in `MANIFEST.JSON` exactly
      (case-sensitive).
- [ ] `META-INF/MANIFEST.JSON` exists inside the bundle and is valid JSON.
- [ ] All required fields (`package`, `name`, `version`, `description`, `author`,
      `entry`) are present in `MANIFEST.JSON`.
- [ ] The `entry` path in `MANIFEST.JSON` points to an existing file inside the bundle.
- [ ] You have restarted the OS (soft-reset or power cycle) after installing.

---

## App crashes on launch

1. **Check the serial console** — connect with `mpremote repl` or Thonny and look for
   a Python traceback.
2. **Common errors:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: no module named 'mpos'` | Running outside MicroPythonOS | Only run apps inside the OS environment |
| `ImportError: no module named 'lvgl'` | Same as above, or wrong firmware | Ensure you are using the MicroPythonOS firmware (not plain MicroPython) |
| `AttributeError: 'MainActivity' object has no attribute 'scr'` | Accessing `self.scr` before it is set in `onCreate` | Initialise all instance attributes in `onCreate` or `__init__` |
| `MemoryError` | Device RAM exhausted | Reduce widget count; call `gc.collect()` in `onPause`/`onDestroy` |

---

## LVGL / display issues

| Symptom | Fix |
|---------|-----|
| Screen flickers or tears | Avoid creating/deleting screen objects in a tight loop |
| Touch events not registered | Ensure you call `lv.scr_load(self.scr)` after creating the screen |
| Text renders as boxes (missing font) | The built-in LVGL font may not contain the characters you need; use ASCII-only text or load a custom font |

---

## Getting more diagnostic information

### View the REPL output

```bash
mpremote connect /dev/ttyUSB0 repl
```

Press `Ctrl+C` to interrupt running code.  Tracebacks appear here.

### Check free memory

```python
import gc
gc.collect()
print("Free RAM:", gc.mem_free(), "bytes")
```

### Check filesystem space

```python
import os
stat = os.statvfs('/')
block_size = stat[0]
free_blocks = stat[3]
print("Free space:", block_size * free_blocks, "bytes")
```

### View installed apps

```python
import os
print(os.listdir('/apps'))
```

---

*← [Deployment](deployment.md) | [Resources →](resources.md)*
