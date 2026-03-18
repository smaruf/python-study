# 06 — Autonomous Drone Guide

A fully autonomous drone is more than a vehicle with GPS — it requires sensor fusion,
state estimation, path planning, a reliable command & control link, and robust
failsafe strategies. This guide covers the full autonomy stack from hardware to mission.

---

## 1. Autonomy Levels (SAE-inspired for UAV)

| Level | Description | Human role |
|---|---|---|
| AL-0 | Manual RC | Pilot controls everything |
| AL-1 | Stabilised (Angle / Altitude hold) | Pilot provides velocity commands |
| AL-2 | Position hold + RTH | Pilot provides position commands |
| AL-3 | Waypoint navigation | Pilot defines mission, drone executes |
| AL-4 | Supervised autonomy (detect & avoid) | Operator monitors |
| AL-5 | Full autonomy (BVLOS, dynamic replanning) | Minimal human oversight |

---

## 2. Autonomous Drone System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Ground Control Station (GCS)                    │
│   Mission Planner / QGC / Custom UI                                    │
│   ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────────────┐  │
│   │ Mission  │  │ Telemetry│  │  Video    │  │  C2 link mgmt     │  │
│   │ editor   │  │ display  │  │  stream   │  │  (4G/LTE/SAT)     │  │
│   └──────────┘  └──────────┘  └───────────┘  └────────────────────┘  │
└───────────────────────────────┬────────────────────────────────────────┘
                                │ MAVLink / DroneCAN
                                │ (encrypted at AL-4+)
┌───────────────────────────────▼────────────────────────────────────────┐
│                        Flight Computer                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ Sensor     │  │ State        │  │ Mission / Navigation         │  │
│  │ Fusion     │  │ Estimation   │  │ ┌──────────────────────────┐ │  │
│  │ (EKF3)     │  │ (EKF3 out)   │  │ │ Path planner             │ │  │
│  └──────┬─────┘  └──────┬───────┘  │ │ Waypoint sequencer       │ │  │
│         └───────────────┘          │ │ Geofence enforcement      │ │  │
│  ┌──────────────────────────────┐  │ └──────────────────────────┘ │  │
│  │     Attitude Controller      │  └──────────────────────────────┘  │
│  │  Roll PID | Pitch PID | Yaw  │                                     │
│  └──────────────────────────────┘                                     │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Actuator Layer: Motor mixer → ESC (DSHOT/CAN) / Servos (PWM)   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                │ I2C / SPI / CAN
┌───────────────▼────────────────────────────────────────────────────────┐
│                        Sensor Suite                                    │
│  IMU (×3)  |  GNSS (×2, RTK opt.)  |  Barometer (×2)                 │
│  Magnetometer  |  Airspeed  |  LiDAR/Sonar  |  Optical Flow           │
│  Camera (EO/IR)  |  Battery monitor  |  Current sensor                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sensor Fusion (EKF)

The Extended Kalman Filter (EKF) is the core of autonomous navigation.

### 3.1 States estimated by EKF3 (ArduPilot)

| State | Symbol | Source |
|---|---|---|
| Position (NED) | p_N, p_E, p_D | GNSS + barometer + optical flow |
| Velocity (NED) | v_N, v_E, v_D | GNSS, IMU integration |
| Attitude (quaternion) | q0, q1, q2, q3 | IMU integration + magnetometer |
| Gyro bias | bg_x/y/z | IMU self-calibration |
| Accel bias | ba_x/y/z | IMU self-calibration |
| Wind velocity | wN, wE | Airspeed + GPS |
| Magnetic field | MN, ME, MD | Magnetometer |

### 3.2 EKF Health Indicators

Monitor these in logs to detect fusion problems:

| Indicator | Healthy range | Action if out-of-range |
|---|---|---|
| EKF variance (position) | < 1.0 m² | Check GPS quality, multipath |
| Innovation (velocity) | < 0.5 m/s | Check GPS/IMU alignment |
| Static margin | > 0.05 | Check magnetometer interference |
| GPS horizontal accuracy | < 2.0 m | Wait for better satellite geometry |
| Vibration (clip count) | 0 | Improve vibration isolation |

### 3.3 Redundant IMU Voting (production)

```
Three IMUs output samples every 2.5 ms:
  IMU-0: ICM-42688-P (primary, isolated)
  IMU-1: ICM-42688-P (secondary, hard-mount)
  IMU-2: BMI088       (tertiary, different vendor)

Voting:
  1. Compare |IMU-0 - IMU-1| and |IMU-1 - IMU-2|
  2. If delta > threshold (e.g., 3 deg/s), flag that IMU as faulty
  3. Use median of remaining two (or single if two faulty — LAND)
```

---

## 4. Path Planning

### 4.1 Simple: Straight-line waypoints

```
Mission: A → B → C → D → Home
  At each waypoint: check arrival (distance < acceptance_radius)
  Transition: fly directly to next waypoint at cruise speed
  Altitude: interpolated linearly between waypoints
```

### 4.2 Medium: Dubins paths (fixed-wing)

For fixed-wing aircraft that cannot hover, use Dubins paths to compute
the minimum-turning-radius trajectory between waypoints.

```python
def dubins_path(start, goal, turn_radius):
    """
    Returns a sequence of arc/straight segments.
    Used by ArduPlane for smooth loiter-to-waypoint transitions.
    """
    # Reference: Dubins, 1957; implementation in ArduPilot AP_Navigation
    pass
```

### 4.3 Production: RRT* / A* with dynamic replanning

```
Obstacle map (3D voxel grid or point cloud from LiDAR)
  ↓
A* or RRT* search → optimal collision-free path
  ↓
Path smoothing (B-spline or polynomial)
  ↓
Velocity profile (trapezoidal or jerk-limited)
  ↓
Feed to trajectory tracking controller (MPC or PID)
```

**Libraries:**
- OMPL (Open Motion Planning Library)
- ArduPilot terrain following (SRTM-based)
- PX4 FlightTaskAutoMapper
- Autonomy SDK (DroneKit, MAVSDK)

---

## 5. Detect and Avoid (DAA)

### 5.1 Sensor options

| Sensor | Range | Use |
|---|---|---|
| ADS-B receiver | 100+ km | Manned aircraft detection |
| FLARM | 1–5 km | Glider/UAS cooperative |
| LiDAR 360° (Livox) | 100–200 m | Close-range obstacle |
| Stereo camera | 5–30 m | Vision-based obstacle |
| Radar (24 GHz) | 50–300 m | All-weather, ignores rain |

### 5.2 DAA Decision Logic

```
Threat assessment:
  Compute time-to-collision (TTC) for each tracked object
  If TTC < threshold:
    1. Issue avoidance manoeuvre (bump lateral + altitude)
    2. Resume original track after clear
    3. Log event and notify GCS

TCAS-like logic:
  RA (Resolution Advisory) → immediate avoidance
  TA (Traffic Advisory)    → monitor, prepare avoidance
```

### 5.3 RTCA SC-228 / ASTM F3442

Production DAA must comply with RTCA DO-365 (DAA Minimum Operational
Performance Standards) for BVLOS operations.

---

## 6. Command & Control (C2) Link

### 6.1 Link types

| Type | Latency | Range | Redundancy role |
|---|---|---|---|
| RC (ELRS / CRSF) | <1 ms | 50 km LoS | Manual override always |
| 900 MHz SiK telemetry | 50 ms | 15 km LoS | Primary MAVLink |
| 4G LTE (mobile) | 50–200 ms | Nationwide | BVLOS primary |
| Iridium satellite | 300–500 ms | Global | BVLOS backup |
| 5G (UAS-UTM) | 10–20 ms | Urban | Future standard |

### 6.2 Failsafe Priority (ArduCopter default)

```
RC signal OK
  └─ Normal flight
RC loss > 1 s
  └─ Continue mission (if BVLOS) OR RTL
Battery < 20%
  └─ RTL
Battery < 10%
  └─ Land immediately
GCS link loss (for guided / auto)
  └─ Continue mission
Geofence breach
  └─ RTL
EKF unhealthy
  └─ Stabilise + land
```

### 6.3 MAVLink Security (production)

```
Layer 1: Authentication
  Pre-shared key (PSK) or certificate per vehicle
  MAVLink 2 signing (SHA-256 HMAC, link_id + timestamp)

Layer 2: Confidentiality
  AES-128 or AES-256 payload encryption (over-the-air)
  TLS 1.3 for cloud MAVLink routing (MAVSDK server)

Layer 3: Anti-replay
  Monotonically-increasing timestamp in signed packets
  Reject packets with timestamp delta > 10 s

Layer 4: Key management
  ATECC608 hardware secure element on flight computer
  OTA key rotation every 90 days
```

---

## 7. Autonomous Mission Lifecycle

```
Pre-flight phase:
  1. Power on → self-test (BIT: Built-in Test)
  2. Sensor calibration check (gyro bias, compass consistency)
  3. GPS HDOP < 2.0, satellite count ≥ 8
  4. Battery state > 80%
  5. GCS upload mission + activate geofence
  6. Arm + takeoff (auto-throttle)

Mission phase:
  7. Climb to mission altitude
  8. Fly waypoints (auto mode)
  9. At each waypoint: loiter / payload release / camera trigger
  10. Monitor safety watchdogs continuously

Return phase:
  11. Mission complete → RTL or new mission upload
  12. Land at home or designated recovery zone
  13. Touchdown detection → disarm

Post-flight phase:
  14. Download logs (MAVLink FTP or USB)
  15. Battery swap
  16. Pre-flight for next sortie
```

---

## 8. Companion Computer Integration

### 8.1 Role split

| Subsystem | Flight Controller (FC) | Companion Computer |
|---|---|---|
| Attitude control | Yes | No |
| Navigation | Yes | Assists (offboard) |
| GNSS fusion | Yes | No |
| Computer vision | No | Yes |
| Payload control | Partial | Yes |
| AI/ML inference | No | Yes |
| LTE / cloud link | No | Yes |
| Logging (high-res) | FC logs | Full mission logs |

### 8.2 Offboard control (MAVSDK Python)

```python
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)

async def fly_offboard(drone):
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, -5.0, 0.0))
    await drone.offboard.start()
    await drone.offboard.set_position_ned(PositionNedYaw(5.0, 0.0, -5.0, 90.0))
    await asyncio.sleep(4)
    await drone.offboard.stop()
```

### 8.3 ROS2 + PX4 integration

```bash
# Install micro-XRCE-DDS agent
sudo snap install micro-xrce-dds-agent

# Start agent (FC connected via UART)
MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600

# ROS2 package for PX4
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
colcon build
source install/setup.bash

# Subscribe to vehicle odometry
ros2 topic echo /fmu/out/vehicle_odometry
```

---

## 9. Testing Strategy for Autonomous Drones

| Test type | Tool | What it validates |
|---|---|---|
| Unit test | pytest (Python), Unity (C) | PID, mixer, MAVLink encoding |
| SITL (Software-in-the-loop) | ArduPilot SITL + Gazebo | Full mission logic, failsafes |
| HITL (Hardware-in-the-loop) | FC + simulated sensors | Real MCU timing, real telemetry |
| Bench test | Tethered at 10 cm | Motor direction, ESC response |
| Ground test | Tethered outdoors | GPS acquisition, compass, RTH |
| Maiden flight | Close-in, low altitude | Trim, hover stability |
| Acceptance test | Full mission profile | End-to-end mission execution |

---

## 10. References

- ArduPilot EKF3 documentation: <https://ardupilot.org/dev/docs/ekf2-navigation-system.html>
- PX4 offboard control: <https://docs.px4.io/main/en/ros/offboard_control.html>
- MAVSDK mission API: <https://mavsdk.mavlink.io/main/en/python/guide/missions.html>
- MAVLink signing: <https://mavlink.io/en/guide/message_signing.html>
- RTCA DO-365 (DAA MOPS): <https://www.rtca.org>
- ASTM F3548 (UTM): <https://www.astm.org>
- Dubins path explanation: <https://en.wikipedia.org/wiki/Dubins_path>
- ROS2 PX4 integration: <https://docs.px4.io/main/en/ros/ros2_comm.html>
