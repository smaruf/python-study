// rc_basic.go — Simple RC firmware skeleton (TinyGo, RP2040 Pico)
// Complexity level: SIMPLE
// Language:         TinyGo
//
// Wiring:
//   GP15 → Servo 1 (aileron)  PWM slice 7
//   GP16 → Servo 2 (elevator) PWM slice 0
//   GP17 → ESC throttle       PWM slice 0 (channel B)
//
// Flash instructions:
//   tinygo flash -target=pico rc_basic.go
//
// Notes:
//   TinyGo goroutines on RP2040 run co-operatively (no pre-emption by default).
//   Use time.Sleep to yield between goroutines.

package main

import (
	"machine"
	"time"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	pwmFreqHz  = 50          // Servo / ESC 50 Hz
	pwmMinUs   = 1_000       // 1000 µs = minimum (full reverse / full left)
	pwmMaxUs   = 2_000       // 2000 µs = maximum (full forward / full right)
	pwmMidUs   = 1_500       // 1500 µs = centre / arm
	periodUs   = 1_000_000 / pwmFreqHz // 20 000 µs period
	loopDelayMs = 20         // main loop 50 Hz
)

// ---------------------------------------------------------------------------
// PWM output helpers
// ---------------------------------------------------------------------------

// initPWM configures a GPIO pin as a 50 Hz PWM output.
// Returns the PWM channel handle.
func initPWM(pin machine.Pin) (machine.PWM, machine.PWMChannel) {
	pwm := machine.PWMFor(pin)
	err := pwm.Configure(machine.PWMConfig{
		Period: uint64(periodUs) * 1_000, // period in nanoseconds
	})
	if err != nil {
		// On failure, hang with LED indicator
		led := machine.LED
		led.Configure(machine.PinConfig{Mode: machine.PinOutput})
		for {
			led.Toggle()
			time.Sleep(200 * time.Millisecond)
		}
	}
	ch, _ := pwm.Channel(pin)
	setPulse(pwm, ch, pwmMidUs)
	return pwm, ch
}

// setPulse sets the PWM output pulse width in microseconds (clamped).
func setPulse(pwm machine.PWM, ch machine.PWMChannel, pulseUs uint32) {
	if pulseUs < pwmMinUs {
		pulseUs = pwmMinUs
	}
	if pulseUs > pwmMaxUs {
		pulseUs = pwmMaxUs
	}
	// duty cycle = pulseUs / periodUs as a fraction of Top
	top := pwm.Top()
	duty := uint32(top) * pulseUs / uint32(periodUs)
	pwm.Set(ch, duty)
}

// ---------------------------------------------------------------------------
// Simulated RC input (replace with actual SBUS decoder or PIO PWM reader)
// ---------------------------------------------------------------------------

// rcChannels holds the latest channel values (µs).
type rcChannels struct {
	throttle uint32
	aileron  uint32
	elevator uint32
}

// readRC returns simulated neutral RC values.
// In a real build, replace this with an SBUS or PWM reader.
func readRC() rcChannels {
	return rcChannels{
		throttle: pwmMinUs, // idle throttle
		aileron:  pwmMidUs,
		elevator: pwmMidUs,
	}
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	// Initialise PWM outputs
	pwmAileron, chAileron   := initPWM(machine.GPIO15)
	pwmElevator, chElevator := initPWM(machine.GPIO16)
	pwmESC, chESC           := initPWM(machine.GPIO17)

	// ESC arming: hold 1000 µs for 2 seconds
	setPulse(pwmESC, chESC, pwmMinUs)
	time.Sleep(2 * time.Second)

	for {
		rc := readRC()

		// Pass-through: no stabilisation at simple level
		setPulse(pwmESC,      chESC,      rc.throttle)
		setPulse(pwmAileron,  chAileron,  rc.aileron)
		setPulse(pwmElevator, chElevator, rc.elevator)

		time.Sleep(loopDelayMs * time.Millisecond)
	}
}
