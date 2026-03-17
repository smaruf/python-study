"""
Microcontroller Firmware & Component Hardware Analysis

Prints complete hardware design, component BOMs, ASCII wiring diagrams,
Raspberry Pi Pico MicroPython firmware, and Arduino C++ firmware for
all three RC aircraft builds:

  1. RCMakerLab Flying-Wing RC Plane  (Oct 2025)
  2. 3JWings DC Motor Stick Plane     (Jan 2025)
  3. Shahed / Lucas Drone study model (educational only)

Run with:
    PYTHONPATH=. python examples/microcontroller_analysis.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fixed_wing.microcontroller_firmware import (
    component_bom_flying_wing,
    component_bom_stick_plane,
    component_bom_shahed_study,
    wiring_diagram_arduino,
    wiring_diagram_pico,
    pico_flying_wing_firmware,
    pico_stick_plane_firmware,
    pico_shahed_study_firmware,
    arduino_flying_wing_full,
    arduino_stick_plane_full,
    arduino_shahed_study_full,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title, width=80, char="="):
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print()


def _sub(title):
    print(f"\n--- {title} ---")


def _bom_table(bom_list):
    """Print a BOM list as a readable table."""
    for item in bom_list:
        ref   = item.get("ref", "")
        qty   = item.get("qty", "")
        part  = item.get("part", "")
        notes = item.get("notes", "")
        print(f"  [{qty}×] {ref:<6} {part:<46}  {notes}")


def _firmware_summary(fw_dict, label):
    """Print firmware platform, libraries, and flash instructions."""
    print(f"  Platform      : {fw_dict.get('platform', fw_dict.get('language', 'N/A'))}")
    if "micropython_library" in fw_dict:
        print(f"  Library       : {fw_dict['micropython_library']}")
    if "libraries_required" in fw_dict:
        print(f"  Libraries     :")
        for lib in fw_dict["libraries_required"]:
            print(f"    - {lib}")
    print(f"  Flash steps   :")
    for step in fw_dict["flash_instructions"]:
        print(f"    {step}")
    tx_key = "transmitter_firmware" if "transmitter_firmware" in fw_dict else "transmitter_sketch"
    rx_key = "receiver_firmware"    if "receiver_firmware"    in fw_dict else "receiver_sketch"
    lines_tx = fw_dict[tx_key].count("\n")
    lines_rx = fw_dict[rx_key].count("\n")
    print(f"  TX firmware   : {lines_tx} lines of code")
    print(f"  RX firmware   : {lines_rx} lines of code")
    print(f"  (access full source via {label}['{tx_key}'] / ['{rx_key}'])")


# ---------------------------------------------------------------------------
# 1. Component BOMs
# ---------------------------------------------------------------------------

def show_bom_flying_wing():
    _header("COMPONENT BOM — RCMakerLab Flying-Wing RC Plane")
    bom = component_bom_flying_wing()

    _sub("Transmitter BOM")
    _bom_table(bom["transmitter_bom"])

    _sub("Receiver BOM")
    _bom_table(bom["receiver_bom"])

    _sub("Shared Airframe / Propulsion BOM")
    for item in bom["shared_bom"]:
        print(f"  [{item['qty']}×] {item['part']:<50}  {item['notes']}")

    _sub("Power Budget (mA)")
    for k, v in bom["power_budget_ma"].items():
        print(f"  {k:<40}: {v}")

    _sub("Design Notes")
    for note in bom["design_notes"]:
        print(f"  ► {note}")


def show_bom_stick_plane():
    _header("COMPONENT BOM — 3JWings DC Motor Stick Plane")
    bom = component_bom_stick_plane()

    _sub("Transmitter BOM")
    _bom_table(bom["transmitter_bom"])

    _sub("Receiver BOM")
    _bom_table(bom["receiver_bom"])

    _sub("Shared Airframe / Propulsion BOM")
    for item in bom["shared_bom"]:
        print(f"  [{item['qty']}×] {item['part']:<50}  {item['notes']}")

    _sub("Power Budget (mA)")
    for k, v in bom["power_budget_ma"].items():
        print(f"  {k:<40}: {v}")

    _sub("Design Notes")
    for note in bom["design_notes"]:
        print(f"  ► {note}")


def show_bom_shahed():
    _header("COMPONENT BOM — Shahed / Lucas Drone Study Model")
    bom = component_bom_shahed_study()
    print(f"  ⚠ {bom['build']}")

    _sub("Transmitter BOM")
    _bom_table(bom["transmitter_bom"])

    _sub("Receiver BOM")
    _bom_table(bom["receiver_bom"])

    _sub("Shared Airframe / Propulsion BOM")
    for item in bom["shared_bom"]:
        print(f"  [{item['qty']}×] {item['part']:<50}  {item['notes']}")

    _sub("Power Budget (mA)")
    for k, v in bom["power_budget_ma"].items():
        print(f"  {k:<40}: {v}")

    _sub("Design Notes")
    for note in bom["design_notes"]:
        print(f"  ► {note}")


# ---------------------------------------------------------------------------
# 2. ASCII Wiring Diagrams
# ---------------------------------------------------------------------------

def show_wiring_diagrams():
    _header("ASCII WIRING DIAGRAMS — ARDUINO")
    ard = wiring_diagram_arduino()
    print(ard["flying_wing"])
    print(ard["stick_plane"])
    print(ard["shahed_study"])

    _header("ASCII WIRING DIAGRAMS — RASPBERRY PI PICO")
    pico = wiring_diagram_pico()
    print(pico["flying_wing"])
    print(pico["stick_plane"])
    print(pico["shahed_study"])


# ---------------------------------------------------------------------------
# 3. RPi Pico MicroPython Firmware Summaries
# ---------------------------------------------------------------------------

def show_pico_firmware():
    _header("RASPBERRY PI PICO — MicroPython FIRMWARE")

    _sub("Flying-Wing RC Plane")
    fw = pico_flying_wing_firmware()
    _firmware_summary(fw, "pico_flying_wing_firmware()")

    _sub("DC Motor Stick Plane")
    fw = pico_stick_plane_firmware()
    _firmware_summary(fw, "pico_stick_plane_firmware()")

    _sub("Shahed / Lucas Study Model")
    fw = pico_shahed_study_firmware()
    if "warning" in fw:
        print(f"  {fw['warning']}")
    _firmware_summary(fw, "pico_shahed_study_firmware()")


# ---------------------------------------------------------------------------
# 4. Arduino C++ Firmware Summaries
# ---------------------------------------------------------------------------

def show_arduino_firmware():
    _header("ARDUINO NANO V3 — Full C++ Firmware (with MPU-6050 IMU)")

    _sub("Flying-Wing RC Plane")
    fw = arduino_flying_wing_full()
    _firmware_summary(fw, "arduino_flying_wing_full()")

    _sub("DC Motor Stick Plane")
    fw = arduino_stick_plane_full()
    _firmware_summary(fw, "arduino_stick_plane_full()")

    _sub("Shahed / Lucas Study Model")
    fw = arduino_shahed_study_full()
    if "warning" in fw:
        print(f"  {fw['warning']}")
    _firmware_summary(fw, "arduino_shahed_study_full()")


# ---------------------------------------------------------------------------
# 5. Platform Comparison
# ---------------------------------------------------------------------------

def show_platform_comparison():
    _header("PLATFORM COMPARISON: Arduino Nano vs Raspberry Pi Pico")
    rows = [
        ("Feature",              "Arduino Nano V3",         "Raspberry Pi Pico"),
        ("CPU",                  "ATmega328P 16 MHz 8-bit", "RP2040 133 MHz 32-bit"),
        ("RAM",                  "2 KB SRAM",               "264 KB SRAM"),
        ("Flash",                "32 KB",                   "2 MB"),
        ("Language",             "C++ (Arduino IDE)",       "MicroPython / C++"),
        ("PWM channels",         "6 × 8-bit",               "16 × 16-bit (hardware)"),
        ("ADC resolution",       "10-bit (0–1023)",         "12-bit (0–4095)"),
        ("I2C / SPI",            "1× each (hardware)",      "2× I2C + 2× SPI"),
        ("Cost (approx)",        "~$3 USD",                 "~$4 USD"),
        ("Servo/ESC PWM",        "Servo library (8-bit)",   "duty_ns() 16-bit (precise)"),
        ("USB Serial",           "CH340 / FT232 chip",      "Built-in USB CDC"),
        ("Boot time",            "< 0.1 s",                 "< 0.5 s (MicroPython)"),
        ("IDE",                  "Arduino IDE 2.x",         "Thonny / mpremote"),
        ("Best for",             "Quick builds, C++ libs",  "Advanced control, precision PWM"),
    ]
    col_w = [28, 30, 30]
    sep = "  " + "-" * (sum(col_w) + 4)
    print(sep)
    for i, row in enumerate(rows):
        line = "  " + "  ".join(f"{row[j]:<{col_w[j]}}" for j in range(3))
        print(line)
        if i == 0:
            print(sep)
    print(sep)

    print()
    print("  Recommendation:")
    print("  • Beginners / C++ experience  → Arduino Nano (more tutorials available)")
    print("  • Precise servo timing        → RPi Pico (16-bit duty_ns vs 8-bit analogWrite)")
    print("  • Future autopilot expansion  → RPi Pico (more RAM for MAVLink / GPS parsing)")
    print("  • Lowest cost build           → Either (< $1 price difference)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print()
    print("█" * 80)
    print("  MICROCONTROLLER FIRMWARE & COMPONENT HARDWARE DESIGN")
    print("  Raspberry Pi Pico (MicroPython) + Arduino Nano (C++)")
    print("  From-scratch component BOMs + wiring + full firmware for 3 builds")
    print("█" * 80)

    # Component BOMs
    show_bom_flying_wing()
    show_bom_stick_plane()
    show_bom_shahed()

    # Wiring diagrams
    show_wiring_diagrams()

    # Firmware summaries (full source available via function calls)
    show_pico_firmware()
    show_arduino_firmware()

    # Platform comparison
    show_platform_comparison()

    _header("ANALYSIS COMPLETE")
    print("  To access full firmware source code:")
    print()
    print("  from fixed_wing.microcontroller_firmware import (")
    print("      pico_flying_wing_firmware,")
    print("      pico_stick_plane_firmware,")
    print("      pico_shahed_study_firmware,")
    print("      arduino_flying_wing_full,")
    print("      arduino_stick_plane_full,")
    print("      arduino_shahed_study_full,")
    print("  )")
    print()
    print("  # Get TX + RX source as strings (copy to .py / .ino and flash):")
    print("  fw = pico_flying_wing_firmware()")
    print("  print(fw['transmitter_firmware'])   # copy to tx/main.py on Pico")
    print("  print(fw['receiver_firmware'])       # copy to rx/main.py on Pico")
    print()
    print("  fw = arduino_flying_wing_full()")
    print("  print(fw['transmitter_sketch'])     # copy to tx.ino in Arduino IDE")
    print("  print(fw['receiver_sketch'])        # copy to rx.ino in Arduino IDE")
    print()


if __name__ == "__main__":
    main()
