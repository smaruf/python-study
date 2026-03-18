"""
rc_basic.py — Simple RC firmware skeleton (MicroPython, RP2040 / Pico)
Complexity level: SIMPLE
Language:         MicroPython

Wiring:
  GP10 ← SBUS / PWM signal from RC receiver (CH1 = throttle)
  GP11 ← CH2 (aileron)
  GP12 ← CH3 (elevator)
  GP15 → Servo 1 (aileron)
  GP16 → Servo 2 (elevator)
  GP17 → ESC signal (throttle PWM)

Flash instructions:
  1. Install MicroPython on the Pico:
     https://micropython.org/download/rp2-pico/
  2. Copy this file to the Pico as main.py
  3. Power on — it runs automatically
"""

from machine import Pin, PWM
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PWM_FREQ_HZ   = 50          # Servo / ESC update rate (50 Hz)
PWM_MIN_US    = 1_000       # Minimum pulse width (µs)
PWM_MAX_US    = 2_000       # Maximum pulse width (µs)
PWM_MID_US    = 1_500       # Centre / arm pulse width (µs)

# GPIO pin assignments
PIN_CH1_THROTTLE  = 10
PIN_CH2_AILERON   = 11
PIN_CH3_ELEVATOR  = 12
PIN_SERVO_AILERON  = 15
PIN_SERVO_ELEVATOR = 16
PIN_ESC_THROTTLE   = 17

# ---------------------------------------------------------------------------
# PWM output helpers
# ---------------------------------------------------------------------------

def init_pwm_out(pin_number: int) -> PWM:
    """Initialise a PWM output at 50 Hz for servo / ESC."""
    pwm = PWM(Pin(pin_number, Pin.OUT))
    pwm.freq(PWM_FREQ_HZ)
    # Arm position: 1500 µs centre
    pwm.duty_ns(PWM_MID_US * 1_000)
    return pwm


def set_pulse_us(pwm: PWM, pulse_us: int) -> None:
    """Set PWM output pulse width in microseconds (clamped to 1000–2000)."""
    pulse_us = max(PWM_MIN_US, min(PWM_MAX_US, pulse_us))
    pwm.duty_ns(pulse_us * 1_000)


# ---------------------------------------------------------------------------
# PWM input (simple polling — no interrupts)
# Accurate enough for slow-loop (50 Hz) control surfaces.
# For higher accuracy use PIO state-machine PWM reader.
# ---------------------------------------------------------------------------

def read_pwm_us(pin_number: int, timeout_us: int = 30_000) -> int:
    """
    Measure a PWM pulse width in microseconds from a digital input pin.
    Returns PWM_MID_US (1500) on timeout.
    """
    pin = Pin(pin_number, Pin.IN)

    # Wait for low → high transition (start of pulse)
    deadline = time.ticks_us() + timeout_us
    while pin.value() == 1:
        if time.ticks_diff(deadline, time.ticks_us()) <= 0:
            return PWM_MID_US

    while pin.value() == 0:
        if time.ticks_diff(deadline, time.ticks_us()) <= 0:
            return PWM_MID_US

    rise = time.ticks_us()

    # Wait for high → low transition (end of pulse)
    while pin.value() == 1:
        if time.ticks_diff(deadline, time.ticks_us()) <= 0:
            return PWM_MID_US

    fall = time.ticks_us()
    pulse = time.ticks_diff(fall, rise)

    # Sanity-check: valid servo pulse is 800–2200 µs
    if 800 <= pulse <= 2200:
        return pulse
    return PWM_MID_US


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("RC Basic firmware starting…")

    # Initialise servo and ESC outputs
    servo_aileron  = init_pwm_out(PIN_SERVO_AILERON)
    servo_elevator = init_pwm_out(PIN_SERVO_ELEVATOR)
    esc_throttle   = init_pwm_out(PIN_ESC_THROTTLE)

    # ESC arming sequence: hold 1000 µs for 2 seconds
    set_pulse_us(esc_throttle, PWM_MIN_US)
    time.sleep(2)
    print("ESC armed.")

    while True:
        # --- Read RC channels ---
        throttle_us = read_pwm_us(PIN_CH1_THROTTLE)
        aileron_us  = read_pwm_us(PIN_CH2_AILERON)
        elevator_us = read_pwm_us(PIN_CH3_ELEVATOR)

        # --- Pass-through to outputs (no stabilisation at this level) ---
        set_pulse_us(esc_throttle,   throttle_us)
        set_pulse_us(servo_aileron,  aileron_us)
        set_pulse_us(servo_elevator, elevator_us)

        # 50 Hz loop rate
        time.sleep_ms(20)


if __name__ == "__main__":
    main()
