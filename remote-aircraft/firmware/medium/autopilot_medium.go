// autopilot_medium.go — Medium autopilot firmware skeleton (TinyGo, RP2040)
// Complexity level: MEDIUM
// Language:         TinyGo
//
// Features:
//   - Goroutine-per-sensor concurrency model
//   - IMU reading (MPU-6050 over I2C)
//   - PID attitude stabilisation
//   - DSHOT300 motor output placeholder (use PIO for real DSHOT)
//   - MAVLink HEARTBEAT via UART0
//   - Arming state machine
//
// Flash:
//   tinygo flash -target=pico autopilot_medium.go
//
// Note: TinyGo goroutines are co-operative; call time.Sleep to yield.

package main

import (
	"machine"
	"time"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	pwmMinUs    uint32 = 1_000
	pwmMaxUs    uint32 = 2_000
	pwmMidUs    uint32 = 1_500
	loopPeriod         = 2500 * time.Microsecond // 400 Hz
	hbPeriod           = 1 * time.Second
)

// Motor GPIO pins (quad X)
var motorPins = [4]machine.Pin{
	machine.GPIO15,
	machine.GPIO16,
	machine.GPIO17,
	machine.GPIO18,
}

// ---------------------------------------------------------------------------
// PID controller
// ---------------------------------------------------------------------------

type PID struct {
	Kp, Ki, Kd     float32
	IntegralLimit  float32
	integral       float32
	prevError      float32
}

func (p *PID) Reset() {
	p.integral  = 0
	p.prevError = 0
}

func (p *PID) Update(setpoint, measured, dt float32) float32 {
	err := setpoint - measured
	p.integral += err * dt
	switch {
	case p.integral >  p.IntegralLimit:
		p.integral =  p.IntegralLimit
	case p.integral < -p.IntegralLimit:
		p.integral = -p.IntegralLimit
	}
	var deriv float32
	if dt > 0 {
		deriv = (err - p.prevError) / dt
	}
	p.prevError = err
	return p.Kp*err + p.Ki*p.integral + p.Kd*deriv
}

// ---------------------------------------------------------------------------
// Minimal MPU-6050 driver
// ---------------------------------------------------------------------------

const (
	mpuAddr     = 0x68
	mpuRegPwr   = 0x6B
	mpuRegGyroX = 0x43
)

func mpuInit(bus *machine.I2C) {
	// Wake up: write 0x00 to PWR_MGMT_1
	_ = bus.WriteRegister(mpuAddr, mpuRegPwr, []byte{0x00})
}

func mpuReadGyro(bus *machine.I2C) (gx, gy, gz float32) {
	buf := make([]byte, 6)
	_ = bus.ReadRegister(mpuAddr, mpuRegGyroX, buf)
	rawX := int16(buf[0])<<8 | int16(buf[1])
	rawY := int16(buf[2])<<8 | int16(buf[3])
	rawZ := int16(buf[4])<<8 | int16(buf[5])
	const scale float32 = 131.0 // ±250 deg/s
	return float32(rawX) / scale, float32(rawY) / scale, float32(rawZ) / scale
}

// ---------------------------------------------------------------------------
// PWM motor output (50 Hz standard ESC)
// ---------------------------------------------------------------------------

type Motors struct {
	pwms [4]machine.PWM
	chs  [4]machine.PWMChannel
}

func initMotors() Motors {
	var m Motors
	for i, pin := range motorPins {
		pwm := machine.PWMFor(pin)
		_ = pwm.Configure(machine.PWMConfig{Period: 20_000_000}) // 20 ms
		ch, _ := pwm.Channel(pin)
		pwm.Set(ch, motorDuty(pwm, pwmMinUs))
		m.pwms[i] = pwm
		m.chs[i]  = ch
	}
	return m
}

func motorDuty(pwm machine.PWM, pulseUs uint32) uint32 {
	if pulseUs < pwmMinUs { pulseUs = pwmMinUs }
	if pulseUs > pwmMaxUs { pulseUs = pwmMaxUs }
	return pwm.Top() * pulseUs / 20_000
}

func (m *Motors) Set(idx int, pulseUs uint32) {
	m.pwms[idx].Set(m.chs[idx], motorDuty(m.pwms[idx], pulseUs))
}

func (m *Motors) SetAll(pulseUs uint32) {
	for i := range motorPins {
		m.Set(i, pulseUs)
	}
}

// ---------------------------------------------------------------------------
// Quad-X mixer
// ---------------------------------------------------------------------------

func mixQuadX(thr, roll, pitch, yaw float32) [4]uint32 {
	raw := [4]float32{
		thr + roll - pitch - yaw, // front-right  CW
		thr - roll + pitch - yaw, // rear-left    CW
		thr + roll + pitch + yaw, // front-left   CCW
		thr - roll - pitch + yaw, // rear-right   CCW
	}
	var out [4]uint32
	for i, v := range raw {
		us := uint32(1000 + v)
		if us < pwmMinUs { us = pwmMinUs }
		if us > pwmMaxUs { us = pwmMaxUs }
		out[i] = us
	}
	return out
}

// ---------------------------------------------------------------------------
// Arming state
// ---------------------------------------------------------------------------

type armState int

const (
	stateDisarmed armState = iota
	stateArmed
	stateFailsafe
)

// ---------------------------------------------------------------------------
// Shared state (goroutine-safe via channels)
// ---------------------------------------------------------------------------

type imuData struct{ gx, gy, gz float32 }
type rcData  struct{ throttle, roll, pitch, yaw float32 }

// ---------------------------------------------------------------------------
// Goroutines
// ---------------------------------------------------------------------------

// imuTask reads the IMU at 400 Hz and publishes to a channel.
func imuTask(bus *machine.I2C, out chan<- imuData) {
	for {
		gx, gy, gz := mpuReadGyro(bus)
		select {
		case out <- imuData{gx, gy, gz}:
		default: // drop if channel full
		}
		time.Sleep(loopPeriod)
	}
}

// rcTask provides simulated neutral RC; replace with SBUS parser.
func rcTask(out chan<- rcData) {
	for {
		select {
		case out <- rcData{throttle: 0, roll: 0, pitch: 0, yaw: 0}:
		default:
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// mavlinkTask sends HEARTBEAT at 1 Hz.
func mavlinkTask(uart *machine.UART) {
	seq := uint8(0)
	for {
		data := buildHeartbeat(seq)
		uart.Write(data)
		seq++
		time.Sleep(hbPeriod)
	}
}

// ---------------------------------------------------------------------------
// Minimal MAVLink HEARTBEAT builder
// ---------------------------------------------------------------------------

func buildHeartbeat(seq uint8) []byte {
	payload := []byte{
		0, 0, 0, 0, // custom_mode (uint32 LE)
		6,          // type: MAV_TYPE_GCS
		8,          // autopilot: MAV_AUTOPILOT_GENERIC
		0,          // base_mode
		0,          // system_status
		3,          // mavlink version
	}
	header := []byte{uint8(len(payload)), seq, 1, 1, 0} // len, seq, sysid, compid, msgid
	cs := mavlinkCRC(append(header, append(payload, 50)...)) // CRC_EXTRA=50
	pkt := append([]byte{0xFE}, append(header, payload...)...)
	return append(pkt, byte(cs), byte(cs>>8))
}

func mavlinkCRC(data []byte) uint16 {
	crc := uint16(0xFFFF)
	for _, b := range data {
		tmp := b ^ uint8(crc&0xFF)
		tmp ^= tmp << 4
		crc = (crc >> 8) ^ (uint16(tmp) << 8) ^ (uint16(tmp) << 3) ^ (uint16(tmp) >> 4)
	}
	return crc
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	// Peripherals
	i2c := machine.I2C0
	_ = i2c.Configure(machine.I2CConfig{
		Frequency: 400_000,
		SDA:       machine.GPIO4,
		SCL:       machine.GPIO5,
	})
	mpuInit(i2c)

	uart := machine.UART0
	_ = uart.Configure(machine.UARTConfig{
		BaudRate: 57600,
		TX:       machine.GPIO0,
		RX:       machine.GPIO1,
	})

	motors := initMotors()
	motors.SetAll(pwmMinUs) // arm: 2 s at 1000 µs
	time.Sleep(2 * time.Second)

	// PIDs
	rollPID  := &PID{Kp: 0.8, Ki: 0.02, Kd: 0.12, IntegralLimit: 300}
	pitchPID := &PID{Kp: 0.9, Ki: 0.02, Kd: 0.14, IntegralLimit: 300}
	yawPID   := &PID{Kp: 0.5, Ki: 0.01, Kd: 0.00, IntegralLimit: 200}

	// Channels
	imuCh := make(chan imuData, 2)
	rcCh  := make(chan rcData, 2)

	go imuTask(i2c, imuCh)
	go rcTask(rcCh)
	go mavlinkTask(uart)

	const dt float32 = 1.0 / 400.0
	state := stateDisarmed
	_ = state // arming logic omitted for brevity; add RC gesture check here

	for {
		var imu imuData
		var rc  rcData
		select {
		case imu = <-imuCh:
		default:
		}
		select {
		case rc = <-rcCh:
		default:
		}

		rollOut  := rollPID.Update(rc.roll,  imu.gx, dt)
		pitchOut := pitchPID.Update(rc.pitch, imu.gy, dt)
		yawOut   := yawPID.Update(rc.yaw,  imu.gz, dt)

		motorUs := mixQuadX(rc.throttle, rollOut, pitchOut, yawOut)
		for i, us := range motorUs {
			motors.Set(i, us)
		}

		time.Sleep(loopPeriod)
	}
}
