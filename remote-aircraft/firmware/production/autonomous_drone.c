/*
 * autonomous_drone.c — Production autonomous drone firmware skeleton (bare-metal C)
 * Complexity level:  PRODUCTION
 * Language:          C (MISRA-C:2023 inspired, no dynamic allocation)
 *
 * This file represents the *flight management layer* running on a high-integrity
 * flight computer (e.g., Cube Orange+ STM32H7 or custom ARM Cortex-M7 board).
 * It is NOT a complete ArduPilot replacement — it illustrates the architectural
 * patterns used in production flight software.
 *
 * Architecture:
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Sensor fusion (EKF stub)  │  Mission manager  │  Safety monitor │
 *   ├──────────────────────────────────────────────────────────────────┤
 *   │            Attitude controller (cascaded PID)                    │
 *   ├──────────────────────────────────────────────────────────────────┤
 *   │            Motor mixer  │  Actuator output (DShot/PWM)           │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * Key production patterns demonstrated:
 *   - Fixed-size state machines (no malloc)
 *   - Watchdog-compatible main loop
 *   - Redundant sensor voting
 *   - Arming / safety interlock
 *   - MAVLink encode + UART DMA-ready API
 *   - Battery failsafe
 *
 * Compile (ARM Cortex-M7, STM32H743):
 *   arm-none-eabi-gcc -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard \
 *     -mthumb -O2 -std=c11 -Wall -Wextra -Wpedantic \
 *     -DSTM32H743xx \
 *     -o autonomous_drone.elf autonomous_drone.c
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Platform types and compile-time assertions
 * ============================================================ */
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int16_t  s16;
typedef int32_t  s32;
typedef float    f32;

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Compile-time assertion (works in C11) */
#define STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)

STATIC_ASSERT(sizeof(f32) == 4, "float must be 32-bit");

/* ============================================================
 * Constants
 * ============================================================ */
#define NUM_MOTORS          6U     /* hexacopter */
#define NUM_IMU_INSTANCES   3U     /* triple-redundant IMU */
#define MAX_WAYPOINTS       50U
#define LOOP_HZ             400U
#define HEARTBEAT_PERIOD_MS 1000U
#define BATT_FAILSAFE_PCT   20.0f

/* Motor pulse limits */
#define PWM_MIN_US 1000U
#define PWM_MAX_US 2000U
#define PWM_MID_US 1500U

/* ============================================================
 * PID controller (no heap, no function pointers in inner loop)
 * ============================================================ */
typedef struct {
    f32 kp, ki, kd;
    f32 integral;
    f32 prev_error;
    f32 integral_limit;
} PidController;

static void pid_reset(PidController *const p) {
    p->integral   = 0.0f;
    p->prev_error = 0.0f;
}

static f32 pid_update(PidController *const p, f32 setpoint,
                       f32 measured, f32 dt) {
    f32 error = setpoint - measured;
    p->integral += error * dt;
    if (p->integral >  p->integral_limit) { p->integral =  p->integral_limit; }
    if (p->integral < -p->integral_limit) { p->integral = -p->integral_limit; }
    f32 deriv = (dt > 0.0f) ? ((error - p->prev_error) / dt) : 0.0f;
    p->prev_error = error;
    return (p->kp * error) + (p->ki * p->integral) + (p->kd * deriv);
}

/* ============================================================
 * IMU sample + redundancy voting
 * ============================================================ */
typedef struct { f32 gx, gy, gz; } ImuSample;

/** Median-of-three vote on a single axis (fault tolerance). */
static f32 vote_axis(f32 a, f32 b, f32 c) {
    if ((a <= b && b <= c) || (c <= b && b <= a)) { return b; }
    if ((b <= a && a <= c) || (c <= a && a <= b)) { return a; }
    return c;
}

static ImuSample vote_imu(const ImuSample sensors[NUM_IMU_INSTANCES]) {
    ImuSample out;
    out.gx = vote_axis(sensors[0].gx, sensors[1].gx, sensors[2].gx);
    out.gy = vote_axis(sensors[0].gy, sensors[1].gy, sensors[2].gy);
    out.gz = vote_axis(sensors[0].gz, sensors[1].gz, sensors[2].gz);
    return out;
}

/* ============================================================
 * Mission manager
 * ============================================================ */
typedef struct {
    f32 lat, lon, alt_m;
    f32 speed_ms;
    bool release_payload;
} Waypoint;

typedef enum {
    MISSION_IDLE = 0,
    MISSION_RUNNING,
    MISSION_COMPLETE,
    MISSION_ABORTED,
} MissionState;

typedef struct {
    Waypoint    waypoints[MAX_WAYPOINTS];
    u8          count;
    u8          current_idx;
    MissionState state;
} MissionManager;

static void mission_init(MissionManager *const m) {
    memset(m, 0, sizeof(*m));
    m->state = MISSION_IDLE;
}

static bool mission_add_waypoint(MissionManager *const m, Waypoint wp) {
    if (m->count >= MAX_WAYPOINTS) { return false; }
    m->waypoints[m->count++] = wp;
    return true;
}

static void mission_start(MissionManager *const m) {
    if (m->count > 0U) {
        m->current_idx = 0U;
        m->state       = MISSION_RUNNING;
    }
}

static void mission_advance(MissionManager *const m) {
    m->current_idx++;
    if (m->current_idx >= m->count) {
        m->state = MISSION_COMPLETE;
    }
}

static const Waypoint *mission_current_wp(const MissionManager *const m) {
    if (m->state == MISSION_RUNNING) {
        return &m->waypoints[m->current_idx];
    }
    return NULL;
}

/* ============================================================
 * Safety monitor
 * ============================================================ */
typedef enum {
    SAFE_OK       = 0,
    SAFE_LOW_BATT = 1,
    SAFE_GEOFENCE = 2,
    SAFE_LINK_LOSS = 3,
    SAFE_IMU_FAIL  = 4,
} SafetyFlag;

typedef struct {
    u32 flags;               /* bitmask of SafetyFlag */
    f32 battery_pct;
    u32 last_rc_ms;          /* ms timestamp of last RC packet */
    u32 uptime_ms;
} SafetyMonitor;

static void safety_init(SafetyMonitor *const s) {
    memset(s, 0, sizeof(*s));
    s->battery_pct = 100.0f;
}

static void safety_update(SafetyMonitor *const s, u32 now_ms) {
    s->uptime_ms = now_ms;

    /* Battery failsafe */
    if (s->battery_pct < BATT_FAILSAFE_PCT) {
        s->flags |= (1U << (u32)SAFE_LOW_BATT);
    }

    /* RC link loss: no RC packet for > 1000 ms */
    if ((now_ms - s->last_rc_ms) > 1000U) {
        s->flags |= (1U << (u32)SAFE_LINK_LOSS);
    }
}

static bool safety_rtl_required(const SafetyMonitor *const s) {
    return (s->flags & ((1U << (u32)SAFE_LOW_BATT) |
                        (1U << (u32)SAFE_LINK_LOSS) |
                        (1U << (u32)SAFE_GEOFENCE))) != 0U;
}

/* ============================================================
 * Hexacopter motor mixer (flat X6)
 * ============================================================ */
static void mix_hex_x(f32 thr, f32 roll, f32 pitch, f32 yaw,
                       u16 out[NUM_MOTORS]) {
    /* Flat-X6 geometry, angles 30°/90°/150°/210°/270°/330° */
    static const f32 ROLL_DIR[6]  = { 0.5f, 1.0f,  0.5f, -0.5f, -1.0f, -0.5f};
    static const f32 PITCH_DIR[6] = { 1.0f, 0.0f, -1.0f, -1.0f,  0.0f,  1.0f};
    static const f32 YAW_DIR[6]   = {-1.0f, 1.0f, -1.0f,  1.0f, -1.0f,  1.0f};

    for (u8 i = 0U; i < NUM_MOTORS; i++) {
        f32 m = thr
              + roll  * ROLL_DIR[i]
              + pitch * PITCH_DIR[i]
              + yaw   * YAW_DIR[i];
        s32 us = (s32)PWM_MIN_US + (s32)(m);
        if (us < (s32)PWM_MIN_US) { us = (s32)PWM_MIN_US; }
        if (us > (s32)PWM_MAX_US) { us = (s32)PWM_MAX_US; }
        out[i] = (u16)us;
    }
}

/* ============================================================
 * Motor output stub (replace with DMA-backed timer CCR writes)
 * ============================================================ */
static void motor_set_all(const u16 pulse_us[NUM_MOTORS]) {
    /* In real firmware: write each pulse_us[i] to TIMx->CCRi */
    (void)pulse_us;
}

/* ============================================================
 * MAVLink HEARTBEAT encoder
 * ============================================================ */
#define MAVLINK_STX    0xFEU
#define MAV_SYS_ID     1U
#define MAV_COMP_ID    1U
#define MSG_HEARTBEAT  0U
#define HB_CRC_EXTRA   50U

static u16 mavlink_crc_accumulate(u8 byte, u16 crc) {
    u8 tmp = byte ^ (u8)(crc & 0xFFU);
    tmp ^= (tmp << 4U);
    return (u16)((crc >> 8U)
                 ^ ((u16)tmp << 8U)
                 ^ ((u16)tmp << 3U)
                 ^ ((u16)tmp >> 4U));
}

static u8 mavlink_seq = 0U;

/** Encode HEARTBEAT into caller-supplied buffer (min 17 bytes).
 *  Returns number of bytes written. */
static u8 encode_heartbeat(u8 *buf, u8 buf_len) {
    if (buf_len < 17U) { return 0U; }

    u8 payload[9] = {0};
    payload[4] = 6U;  /* MAV_TYPE_HEXAROTOR */
    payload[5] = 3U;  /* MAV_AUTOPILOT_ARDUPILOTMEGA */
    payload[8] = 3U;  /* MAVLink version 3 */

    u8 header[5] = {
        (u8)sizeof(payload),
        mavlink_seq++,
        MAV_SYS_ID,
        MAV_COMP_ID,
        MSG_HEARTBEAT
    };

    /* Compute CRC */
    u16 crc = 0xFFFFU;
    for (u8 i = 0U; i < (u8)sizeof(header); i++) {
        crc = mavlink_crc_accumulate(header[i], crc);
    }
    for (u8 i = 0U; i < (u8)sizeof(payload); i++) {
        crc = mavlink_crc_accumulate(payload[i], crc);
    }
    crc = mavlink_crc_accumulate(HB_CRC_EXTRA, crc);

    /* Assemble packet */
    u8 idx = 0U;
    buf[idx++] = MAVLINK_STX;
    for (u8 i = 0U; i < (u8)sizeof(header);  i++) { buf[idx++] = header[i];  }
    for (u8 i = 0U; i < (u8)sizeof(payload); i++) { buf[idx++] = payload[i]; }
    buf[idx++] = (u8)(crc & 0xFFU);
    buf[idx++] = (u8)(crc >> 8U);
    return idx;
}

/* ============================================================
 * SysTick (1 ms) — stub; replace with MCU vector
 * ============================================================ */
static volatile u32 g_tick_ms = 0U;
void SysTick_Handler(void) { g_tick_ms++; }

static void delay_ms(u32 ms) {
    u32 end = g_tick_ms + ms;
    while (g_tick_ms < end) { /* wait */ }
}

/* ============================================================
 * UART transmit stub
 * ============================================================ */
static void uart_write(const u8 *data, u8 len) {
    /* Replace with DMA UART write */
    (void)data; (void)len;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    /* ---- Hardware init (stubbed) ---- */
    /* systick_init();  i2c_init();  uart_init();  timer_init(); */

    /* ESC arming */
    u16 arm_pulse[NUM_MOTORS];
    for (u8 i = 0U; i < NUM_MOTORS; i++) { arm_pulse[i] = PWM_MIN_US; }
    motor_set_all(arm_pulse);
    delay_ms(2000U);

    /* ---- PID init ---- */
    PidController roll_pid  = {.kp=1.2f, .ki=0.03f, .kd=0.15f, .integral_limit=400.f};
    PidController pitch_pid = {.kp=1.2f, .ki=0.03f, .kd=0.15f, .integral_limit=400.f};
    PidController yaw_pid   = {.kp=0.8f, .ki=0.01f, .kd=0.00f, .integral_limit=200.f};
    pid_reset(&roll_pid); pid_reset(&pitch_pid); pid_reset(&yaw_pid);

    /* ---- Mission setup ---- */
    MissionManager mission;
    mission_init(&mission);
    mission_add_waypoint(&mission, (Waypoint){.lat=47.39f,.lon=8.54f,.alt_m=30.f,.speed_ms=12.f});
    mission_add_waypoint(&mission, (Waypoint){.lat=47.40f,.lon=8.54f,.alt_m=30.f,.speed_ms=12.f,.release_payload=true});
    mission_add_waypoint(&mission, (Waypoint){.lat=47.41f,.lon=8.54f,.alt_m=30.f,.speed_ms=12.f});
    mission_start(&mission);

    /* ---- Safety init ---- */
    SafetyMonitor safety;
    safety_init(&safety);

    /* ---- Control loop ---- */
    const f32 dt = 1.0f / (f32)LOOP_HZ;
    u32 hb_timer = 0U;
    u8  hb_buf[20];

    while (1) {
        u32 now = g_tick_ms;

        /* Read IMU (stub: all zeros) */
        ImuSample raw_imu[NUM_IMU_INSTANCES] = {{0}};
        ImuSample imu = vote_imu(raw_imu);

        /* RC setpoints (stub: level hover) */
        f32 rc_thr = 0.0f, rc_roll = 0.0f, rc_pitch = 0.0f, rc_yaw = 0.0f;
        safety.last_rc_ms = now;          /* update in real ISR */

        /* Safety check */
        safety_update(&safety, now);
        if (safety_rtl_required(&safety)) {
            /* Override setpoints to RTL behaviour */
            rc_thr  = 500.0f;
            rc_roll = rc_pitch = rc_yaw = 0.0f;
        }

        /* Attitude PID */
        f32 roll_out  = pid_update(&roll_pid,  rc_roll,  imu.gx, dt);
        f32 pitch_out = pid_update(&pitch_pid, rc_pitch, imu.gy, dt);
        f32 yaw_out   = pid_update(&yaw_pid,   rc_yaw,   imu.gz, dt);

        /* Motor mixer */
        u16 motor_us[NUM_MOTORS];
        mix_hex_x(rc_thr, roll_out, pitch_out, yaw_out, motor_us);
        motor_set_all(motor_us);

        /* Mission waypoint advance (simplified — replace with distance check) */
        if (mission.state == MISSION_RUNNING) {
            const Waypoint *wp = mission_current_wp(&mission);
            if (wp != NULL && wp->release_payload) {
                /* trigger payload release GPIO */
            }
            /* In a real system: advance when within acceptance_radius of wp */
            (void)wp;
        }

        /* Heartbeat at 1 Hz */
        if ((now - hb_timer) >= HEARTBEAT_PERIOD_MS) {
            u8 len = encode_heartbeat(hb_buf, (u8)sizeof(hb_buf));
            uart_write(hb_buf, len);
            hb_timer = now;
        }

        /* Maintain loop rate (watchdog pet goes here) */
        while ((g_tick_ms - now) < (1000U / LOOP_HZ));
    }
    /* never reached */
    return 0;
}
