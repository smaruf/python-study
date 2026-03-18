// autonomous_drone.go — Production autonomous drone firmware skeleton (TinyGo)
// Complexity level:  PRODUCTION
// Language:          TinyGo (goroutine-per-subsystem)
//
// Architecture:
//   goroutine: sensorFusion   — reads 3× IMU, votes best value
//   goroutine: missionManager — tracks waypoints, issues goto commands
//   goroutine: safetyMonitor  — watches battery, geofence, link loss
//   goroutine: mavlinkTx      — sends HEARTBEAT + STATUS_TEXT
//   goroutine: attitudeLoop   — 400 Hz PID + motor output
//
// All inter-goroutine communication uses buffered channels to prevent
// blocking the time-critical attitude loop.
//
// Flash:
//   tinygo flash -target=pico autonomous_drone.go
//
// Note: For production hardware (STM32H7 / Cube), generate a TinyGo target
//       file or use the STM32 family target in TinyGo's target repository.

package main

import (
	"machine"
	"time"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	numMotors       = 6
	loopRate        = 400                             // Hz
	loopPeriod      = time.Second / loopRate          // 2.5 ms
	hbInterval      = time.Second
	battFailsafePct = 20.0
	linkLossTimeout = time.Second
)

// ---------------------------------------------------------------------------
// PID controller
// ---------------------------------------------------------------------------

type PID struct {
	Kp, Ki, Kd    float32
	IntegralLimit float32
	integral      float32
	prevError     float32
}

func (p *PID) Reset() { p.integral = 0; p.prevError = 0 }

func (p *PID) Update(setpoint, measured, dt float32) float32 {
	err := setpoint - measured
	p.integral += err * dt
	switch {
	case p.integral >  p.IntegralLimit: p.integral =  p.IntegralLimit
	case p.integral < -p.IntegralLimit: p.integral = -p.IntegralLimit
	}
	var d float32
	if dt > 0 {
		d = (err - p.prevError) / dt
	}
	p.prevError = err
	return p.Kp*err + p.Ki*p.integral + p.Kd*d
}

// ---------------------------------------------------------------------------
// IMU types and redundancy voting
// ---------------------------------------------------------------------------

type ImuSample struct{ Gx, Gy, Gz float32 }

func median3(a, b, c float32) float32 {
	if (a <= b && b <= c) || (c <= b && b <= a) { return b }
	if (b <= a && a <= c) || (c <= a && a <= b) { return a }
	return c
}

func voteIMU(s [3]ImuSample) ImuSample {
	return ImuSample{
		Gx: median3(s[0].Gx, s[1].Gx, s[2].Gx),
		Gy: median3(s[0].Gy, s[1].Gy, s[2].Gy),
		Gz: median3(s[0].Gz, s[1].Gz, s[2].Gz),
	}
}

// readIMUInstance reads one IMU instance from I2C (stub).
func readIMUInstance(_ uint8) ImuSample {
	return ImuSample{} // replace with I2C MPU-6050 / BMI088 read
}

// ---------------------------------------------------------------------------
// Waypoint / mission
// ---------------------------------------------------------------------------

type Waypoint struct {
	Lat, Lon, AltM  float32
	SpeedMS         float32
	ReleasePayload  bool
}

type MissionStatus int

const (
	MissionIdle     MissionStatus = iota
	MissionRunning
	MissionComplete
	MissionAborted
)

type MissionState struct {
	Waypoints   []Waypoint
	CurrentIdx  int
	Status      MissionStatus
}

func (m *MissionState) Current() *Waypoint {
	if m.Status == MissionRunning && m.CurrentIdx < len(m.Waypoints) {
		return &m.Waypoints[m.CurrentIdx]
	}
	return nil
}

func (m *MissionState) Advance() {
	m.CurrentIdx++
	if m.CurrentIdx >= len(m.Waypoints) {
		m.Status = MissionComplete
	}
}

// ---------------------------------------------------------------------------
// Safety flags
// ---------------------------------------------------------------------------

type SafetyFlags uint32

const (
	SafeLowBatt  SafetyFlags = 1 << 0
	SafeLinkLoss SafetyFlags = 1 << 1
	SafeGeofence SafetyFlags = 1 << 2
	SafeIMUFail  SafetyFlags = 1 << 3
)

type SafetyState struct {
	Flags      SafetyFlags
	BatteryPct float32
	LastRC     time.Time
}

func (s *SafetyState) NeedsRTL() bool {
	return (s.Flags & (SafeLowBatt | SafeLinkLoss | SafeGeofence)) != 0
}

// ---------------------------------------------------------------------------
// Hex-X motor mixer
// ---------------------------------------------------------------------------

var rollDir  = [numMotors]float32{ 0.5,  1.0,  0.5, -0.5, -1.0, -0.5}
var pitchDir = [numMotors]float32{ 1.0,  0.0, -1.0, -1.0,  0.0,  1.0}
var yawDir   = [numMotors]float32{-1.0,  1.0, -1.0,  1.0, -1.0,  1.0}

func mixHexX(thr, roll, pitch, yaw float32) [numMotors]uint32 {
	var out [numMotors]uint32
	for i := 0; i < numMotors; i++ {
		v := thr + roll*rollDir[i] + pitch*pitchDir[i] + yaw*yawDir[i]
		us := uint32(1000 + v)
		if us < 1000 { us = 1000 }
		if us > 2000 { us = 2000 }
		out[i] = us
	}
	return out
}

// ---------------------------------------------------------------------------
// Motor output (PWM)
// ---------------------------------------------------------------------------

var motorPins = [numMotors]machine.Pin{
	machine.GPIO15, machine.GPIO16, machine.GPIO17,
	machine.GPIO18, machine.GPIO19, machine.GPIO20,
}

type Motors struct {
	pwms [numMotors]machine.PWM
	chs  [numMotors]machine.PWMChannel
}

func initMotors() Motors {
	var m Motors
	for i, pin := range motorPins {
		p := machine.PWMFor(pin)
		_ = p.Configure(machine.PWMConfig{Period: 20_000_000})
		ch, _ := p.Channel(pin)
		p.Set(ch, p.Top()*1000/20_000)
		m.pwms[i] = p
		m.chs[i]  = ch
	}
	return m
}

func (m *Motors) Set(idx int, pulseUs uint32) {
	if pulseUs < 1000 { pulseUs = 1000 }
	if pulseUs > 2000 { pulseUs = 2000 }
	m.pwms[idx].Set(m.chs[idx], m.pwms[idx].Top()*pulseUs/20_000)
}

func (m *Motors) SetAll(pulseUs uint32) {
	for i := range motorPins { m.Set(i, pulseUs) }
}

// ---------------------------------------------------------------------------
// MAVLink HEARTBEAT builder
// ---------------------------------------------------------------------------

func mavCRC(data []byte) uint16 {
	crc := uint16(0xFFFF)
	for _, b := range data {
		tmp := b ^ uint8(crc&0xFF)
		tmp ^= tmp << 4
		crc = (crc >> 8) ^ (uint16(tmp) << 8) ^ (uint16(tmp) << 3) ^ (uint16(tmp) >> 4)
	}
	return crc
}

func buildHeartbeat(seq uint8) []byte {
	payload := []byte{0, 0, 0, 0, 13, 3, 0, 4, 3} // type=HEXAROTOR, AP=ArduPilot
	header  := []byte{uint8(len(payload)), seq, 1, 1, 0}
	cs := mavCRC(append(append(header, payload...), 50)) // CRC_EXTRA=50
	pkt := append([]byte{0xFE}, append(header, payload...)...)
	return append(pkt, byte(cs), byte(cs>>8))
}

// ---------------------------------------------------------------------------
// Inter-goroutine channels
// ---------------------------------------------------------------------------

var (
	imuCh     = make(chan ImuSample, 4)
	safetyCh  = make(chan SafetyFlags, 4)
	missionCh = make(chan MissionState, 2)
)

// ---------------------------------------------------------------------------
// Goroutines
// ---------------------------------------------------------------------------

// sensorFusion reads all three IMU instances, votes, and publishes.
func sensorFusion() {
	for {
		var raw [3]ImuSample
		for i := uint8(0); i < 3; i++ {
			raw[i] = readIMUInstance(i)
		}
		voted := voteIMU(raw)
		select {
		case imuCh <- voted:
		default:
		}
		time.Sleep(loopPeriod)
	}
}

// missionManager tracks waypoints and publishes state updates.
func missionManager(state *MissionState) {
	for {
		select {
		case missionCh <- *state:
		default:
		}
		// In production: check distance to current WP → advance when within radius
		time.Sleep(500 * time.Millisecond)
	}
}

// safetyMonitor watches battery and RC link; publishes flags.
func safetyMonitor(s *SafetyState) {
	for {
		s.BatteryPct = readBatteryPct() // stub
		if s.BatteryPct < battFailsafePct {
			s.Flags |= SafeLowBatt
		}
		if time.Since(s.LastRC) > linkLossTimeout {
			s.Flags |= SafeLinkLoss
		}
		select {
		case safetyCh <- s.Flags:
		default:
		}
		time.Sleep(5 * time.Second)
	}
}

// readBatteryPct is a stub — replace with ADC voltage divider read.
func readBatteryPct() float32 { return 100.0 }

// mavlinkTx sends HEARTBEAT at 1 Hz.
func mavlinkTx(uart *machine.UART) {
	seq := uint8(0)
	for {
		uart.Write(buildHeartbeat(seq))
		seq++
		time.Sleep(hbInterval)
	}
}

// ---------------------------------------------------------------------------
// Main — attitude control loop
// ---------------------------------------------------------------------------

func main() {
	// Peripheral init
	i2c := machine.I2C0
	_ = i2c.Configure(machine.I2CConfig{Frequency: 400_000,
		SDA: machine.GPIO4, SCL: machine.GPIO5})

	uart := machine.UART0
	_ = uart.Configure(machine.UARTConfig{
		BaudRate: 115200, TX: machine.GPIO0, RX: machine.GPIO1})

	motors := initMotors()
	motors.SetAll(1000) // arm
	time.Sleep(2 * time.Second)

	// PIDs
	rollPID  := &PID{Kp: 1.2, Ki: 0.03, Kd: 0.15, IntegralLimit: 400}
	pitchPID := &PID{Kp: 1.2, Ki: 0.03, Kd: 0.15, IntegralLimit: 400}
	yawPID   := &PID{Kp: 0.8, Ki: 0.01, Kd: 0.00, IntegralLimit: 200}

	// Mission
	mission := &MissionState{
		Status: MissionRunning,
		Waypoints: []Waypoint{
			{Lat: 47.39, Lon: 8.54, AltM: 30, SpeedMS: 12},
			{Lat: 47.40, Lon: 8.54, AltM: 30, SpeedMS: 12, ReleasePayload: true},
		},
	}

	safety := &SafetyState{BatteryPct: 100.0, LastRC: time.Now()}

	go sensorFusion()
	go missionManager(mission)
	go safetyMonitor(safety)
	go mavlinkTx(uart)

	const dt float32 = 1.0 / float32(loopRate)

	for {
		// Collect latest data (non-blocking)
		var imu ImuSample
		var flags SafetyFlags
		select { case imu = <-imuCh:   default: }
		select { case flags = <-safetyCh: default: }

		rc := struct{ thr, roll, pitch, yaw float32 }{} // stub RC input
		if (flags & (SafeLowBatt | SafeLinkLoss)) != 0 {
			rc.thr = 500.0 // hover throttle for RTL descent
		}

		rollOut  := rollPID.Update(rc.roll,  imu.Gx, dt)
		pitchOut := pitchPID.Update(rc.pitch, imu.Gy, dt)
		yawOut   := yawPID.Update(rc.yaw,  imu.Gz, dt)

		motorUs := mixHexX(rc.thr, rollOut, pitchOut, yawOut)
		for i, us := range motorUs {
			motors.Set(i, us)
		}

		time.Sleep(loopPeriod)
	}
}
