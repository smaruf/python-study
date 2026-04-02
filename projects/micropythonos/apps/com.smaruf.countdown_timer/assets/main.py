"""
Countdown Timer — MicroPythonOS App
=====================================
Demonstrates:
  - mpos.Activity lifecycle (onCreate, onResume, onPause, onDestroy)
  - LVGL widgets: lv.obj (screen), lv.arc, lv.label, lv.btn
  - MicroPython machine.Timer for periodic callbacks
  - Proper resource cleanup on exit

App layout (320 x 480 px display):

  ┌────────────────────────────────┐
  │        Countdown Timer         │  ← title label
  │                                │
  │          ╭────────╮            │
  │          │  60    │            │  ← arc (ring) + seconds label inside
  │          ╰────────╯            │
  │                                │
  │   [  Start  ] [  Reset  ]      │  ← control buttons
  └────────────────────────────────┘
"""

import mpos         # MicroPythonOS framework  (available inside the OS only)
import lvgl as lv   # LVGL Python bindings     (bundled with MicroPythonOS firmware)

try:
    from machine import Timer   # Hardware timer on ESP32
except ImportError:
    # On the Linux desktop simulator 'machine' may not be available;
    # fall back to a simple polling approach via lv.timer_create.
    Timer = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SECONDS = 60          # Initial countdown value
ARC_MAX = DEFAULT_SECONDS     # Arc end value = full circle


class MainActivity(mpos.Activity):
    """Countdown Timer — main (and only) Activity."""

    # ------------------------------------------------------------------
    # Lifecycle: onCreate
    # ------------------------------------------------------------------

    def onCreate(self):
        """Build the UI.  Called once when the app is first launched."""
        self._seconds = DEFAULT_SECONDS
        self._running = False
        self._timer = None

        # --- Screen -------------------------------------------------------
        self.scr = lv.obj()
        self.scr.set_style_bg_color(lv.color_hex(0x1A1A2E), lv.PART.MAIN)
        lv.scr_load(self.scr)

        # --- Title label --------------------------------------------------
        self._title = lv.label(self.scr)
        self._title.set_text("Countdown Timer")
        self._title.set_style_text_color(lv.color_hex(0xE0E0E0), lv.PART.MAIN)
        self._title.set_style_text_font(lv.font_montserrat_20, lv.PART.MAIN)
        self._title.align(lv.ALIGN.TOP_MID, 0, 16)

        # --- Arc (countdown ring) -----------------------------------------
        self._arc = lv.arc(self.scr)
        self._arc.set_size(200, 200)
        self._arc.set_range(0, ARC_MAX)
        self._arc.set_value(ARC_MAX)
        self._arc.set_rotation(270)         # start at top
        self._arc.set_bg_angles(0, 360)     # full background circle
        self._arc.set_style_arc_color(
            lv.color_hex(0x16213E), lv.PART.INDICATOR)
        self._arc.set_style_arc_color(
            lv.color_hex(0x0F3460), lv.PART.MAIN)
        self._arc.set_style_arc_width(12, lv.PART.INDICATOR)
        self._arc.set_style_arc_width(12, lv.PART.MAIN)
        self._arc.remove_style(None, lv.PART.KNOB)  # hide knob
        self._arc.align(lv.ALIGN.CENTER, 0, -30)

        # --- Seconds label (inside arc) -----------------------------------
        self._sec_label = lv.label(self.scr)
        self._sec_label.set_style_text_font(lv.font_montserrat_48, lv.PART.MAIN)
        self._sec_label.set_style_text_color(lv.color_hex(0xE94560), lv.PART.MAIN)
        self._update_seconds_label()
        self._sec_label.align_to(self._arc, lv.ALIGN.CENTER, 0, 0)

        # --- Buttons ------------------------------------------------------
        btn_y = 210    # vertical offset from centre

        self._btn_start = self._make_button("Start", -70, btn_y,
                                            self._on_start_stop)
        self._btn_reset = self._make_button("Reset", 70, btn_y,
                                            self._on_reset)

    # ------------------------------------------------------------------
    # Lifecycle: onResume / onPause / onDestroy
    # ------------------------------------------------------------------

    def onResume(self):
        """App is in the foreground — resume any paused timer."""
        # Nothing to do here unless we want to auto-resume a paused countdown.
        pass

    def onPause(self):
        """App went to background — pause the countdown timer."""
        self._stop_hw_timer()

    def onDestroy(self):
        """App is closing — release all resources."""
        self._stop_hw_timer()
        if self.scr:
            self.scr.delete()
            self.scr = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_button(self, text, x_offset, y_offset, callback):
        """Create a styled button aligned relative to centre."""
        btn = lv.btn(self.scr)
        btn.set_size(120, 48)
        btn.set_style_bg_color(lv.color_hex(0x0F3460), lv.PART.MAIN)
        btn.set_style_radius(8, lv.PART.MAIN)
        btn.align(lv.ALIGN.CENTER, x_offset, y_offset)
        btn.add_event_cb(callback, lv.EVENT.CLICKED, None)

        lbl = lv.label(btn)
        lbl.set_text(text)
        lbl.center()
        return btn

    def _update_seconds_label(self):
        """Refresh the seconds label and the arc value."""
        self._sec_label.set_text(str(self._seconds))
        self._arc.set_value(self._seconds)

    def _start_hw_timer(self):
        """Start a 1-second hardware timer (ESP32) or LVGL timer (desktop)."""
        if Timer is not None:
            # ESP32 hardware timer — period in milliseconds
            self._timer = Timer(0)
            self._timer.init(period=1000, mode=Timer.PERIODIC,
                             callback=self._tick)
        else:
            # Desktop fallback: LVGL periodic timer (period in ms)
            self._timer = lv.timer_create(self._tick_lv, 1000, None)

    def _stop_hw_timer(self):
        """Cancel the running timer, if any."""
        if self._timer is not None:
            if Timer is not None:
                self._timer.deinit()
            else:
                self._timer.del_()
            self._timer = None

    # ------------------------------------------------------------------
    # Timer callbacks
    # ------------------------------------------------------------------

    def _tick(self, _t=None):
        """Called every second by the hardware timer (ESP32)."""
        if self._seconds > 0:
            self._seconds -= 1
            # LVGL must be updated from the main thread;
            # on ESP32 we schedule via lv.task_handler being called in the OS loop.
            self._update_seconds_label()
        if self._seconds == 0:
            self._running = False
            self._stop_hw_timer()
            self._btn_start.get_child(0).set_text("Start")  # reset button label

    def _tick_lv(self, timer):
        """Called every second by the LVGL timer (desktop simulator)."""
        self._tick()

    # ------------------------------------------------------------------
    # Button event callbacks
    # ------------------------------------------------------------------

    def _on_start_stop(self, event):
        """Toggle the countdown between running and paused."""
        if self._running:
            # Pause
            self._running = False
            self._stop_hw_timer()
            self._btn_start.get_child(0).set_text("Start")
        else:
            if self._seconds == 0:
                return  # nothing to do; press Reset first
            # Start
            self._running = True
            self._start_hw_timer()
            self._btn_start.get_child(0).set_text("Stop")

    def _on_reset(self, event):
        """Reset the countdown to the default value."""
        self._running = False
        self._stop_hw_timer()
        self._seconds = DEFAULT_SECONDS
        self._update_seconds_label()
        self._btn_start.get_child(0).set_text("Start")
