# App Development Guide

This guide walks you through building MicroPythonOS apps, from a minimal Hello World
to the fully-featured Countdown Timer example.

---

## Table of Contents

- [App structure overview](#app-structure-overview)
- [MANIFEST.JSON reference](#manifestjson-reference)
- [The `mpos.Activity` lifecycle](#the-mposactivity-lifecycle)
- [Hello World app](#hello-world-app)
- [Countdown Timer app (with LVGL UI)](#countdown-timer-app-with-lvgl-ui)
- [Running on desktop](#running-on-desktop)
- [Tips and best practices](#tips-and-best-practices)

---

## App structure overview

Every MicroPythonOS app is a **bundle** — a folder whose name is the app's
reverse-domain bundle ID (e.g., `com.smaruf.countdown_timer`).

```
com.smaruf.countdown_timer/
├── META-INF/
│   └── MANIFEST.JSON        ← required app metadata
├── assets/
│   └── main.py              ← entry point (required)
└── res/
    └── icon-48.png          ← launcher icon (recommended, 48×48 px PNG)
```

Apps are installed under `/apps/` on the device filesystem:

```
/apps/
└── com.smaruf.countdown_timer/
    ├── META-INF/MANIFEST.JSON
    ├── assets/main.py
    └── res/icon-48.png
```

---

## MANIFEST.JSON reference

```json
{
  "package": "com.smaruf.countdown_timer",
  "name": "Countdown Timer",
  "version": "1.0.0",
  "description": "A simple LVGL countdown timer built for MicroPythonOS.",
  "author": "smaruf",
  "entry": "assets/main.py",
  "min_os_version": "1.0.0",
  "permissions": []
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `package` | ✅ | Unique reverse-domain ID; matches folder name |
| `name` | ✅ | Display name shown in launcher |
| `version` | ✅ | Semantic version string |
| `description` | ✅ | Short description |
| `author` | ✅ | Author name or identifier |
| `entry` | ✅ | Relative path to the Python entry point |
| `min_os_version` | ☑️ optional | Minimum required MicroPythonOS version |
| `permissions` | ☑️ optional | Array of permission strings (e.g. `"wifi"`) |

---

## The `mpos.Activity` lifecycle

Every app entry point must define a class that subclasses `mpos.Activity`.
The OS calls lifecycle methods automatically:

```
App launched
     │
     ▼
  onCreate()        ← Build your UI here; called once
     │
     ▼
  onResume()        ← App is in the foreground (may be called multiple times)
     │
     ▼
  onPause()         ← App goes to background (another app opened, etc.)
     │
     ▼
  onDestroy()       ← App is being closed; clean up resources
```

### Lifecycle hook signatures

```python
import mpos

class MainActivity(mpos.Activity):

    def onCreate(self):
        """Called once when the app is first created."""
        pass

    def onResume(self):
        """Called each time the app becomes the active foreground app."""
        pass

    def onPause(self):
        """Called when the app loses focus (goes to background)."""
        pass

    def onDestroy(self):
        """Called when the app is fully closed; release resources."""
        pass
```

> **Note:** The class **must** be named `MainActivity` unless you specify
> an alternative entry class in `MANIFEST.JSON` (advanced use).

---

## Hello World app

Create the following directory structure:

```
com.smaruf.hello_world/
├── META-INF/
│   └── MANIFEST.JSON
└── assets/
    └── main.py
```

**`META-INF/MANIFEST.JSON`**

```json
{
  "package": "com.smaruf.hello_world",
  "name": "Hello World",
  "version": "1.0.0",
  "description": "Minimal MicroPythonOS app.",
  "author": "smaruf",
  "entry": "assets/main.py"
}
```

**`assets/main.py`**

```python
import mpos
import lvgl as lv


class MainActivity(mpos.Activity):

    def onCreate(self):
        # Create a screen
        self.scr = lv.obj()
        lv.scr_load(self.scr)

        # Centre a label
        label = lv.label(self.scr)
        label.set_text("Hello, MicroPythonOS!")
        label.center()

    def onDestroy(self):
        # Clean up the screen object
        if self.scr:
            self.scr.delete()
            self.scr = None
```

Install it on desktop:

```bash
cp -r com.smaruf.hello_world ~/.micropythonos/internal_filesystem/apps/
# Restart the desktop OS — the app appears in the launcher
```

---

## Countdown Timer app (with LVGL UI)

The fully-annotated example lives at
[`../apps/com.smaruf.countdown_timer/`](../apps/com.smaruf.countdown_timer/).

Key features demonstrated:

- An **arc widget** as a visual countdown ring
- A **label** showing remaining seconds
- **Start / Stop / Reset buttons** with touch callbacks
- A MicroPython **`Timer`** that fires every second
- Proper resource cleanup in `onDestroy`

See [`../apps/com.smaruf.countdown_timer/assets/main.py`](../apps/com.smaruf.countdown_timer/assets/main.py)
for the complete, commented source code.

---

## Running on desktop

1. Install the desktop OS (see [installation.md](installation.md#linux-desktop-install)).
2. Copy your app bundle into the virtual filesystem:

```bash
APP_DIR="apps/com.smaruf.countdown_timer"   # relative to this project
DEST=~/.micropythonos/internal_filesystem/apps/

cp -r "$APP_DIR" "$DEST"
```

Or use the helper script:

```bash
bash ../scripts/install_app.sh --app apps/com.smaruf.countdown_timer --target desktop
```

3. Start the desktop OS:

```bash
cd ~/micropythonos
./micropythonos
```

4. Open the **App Launcher** from the home screen and tap your app.

---

## Tips and best practices

| Tip | Rationale |
|-----|-----------|
| Always clean up in `onDestroy` | Memory leaks are visible on-device since RAM is very limited |
| Avoid blocking calls in lifecycle methods | The OS UI event loop runs in the same thread; use timers or async patterns |
| Use `lv.scr_act()` carefully | Avoid switching screens outside `onCreate`/`onResume` to prevent UI glitches |
| Test on desktop first | Iteration is much faster than re-flashing the ESP32 |
| Keep app assets small | LittleFS partition is typically 4–8 MB on most boards |
| Log with `print()` | Output appears in the REPL / `mpremote` serial console |

---

*← [Installation](installation.md) | [Deployment →](deployment.md)*
