/*
 * esp_blast_flight_controller.c — ESP-BLAST Rocket Drone firmware skeleton
 *
 * Project:  ESP-BLAST (Smallest ESP32 Brushless Rocket Drone)
 * Source:   https://www.instructables.com/Build-the-Smallest-ESP32-Brushless-Rocket-Drone-ES/
 * Language: C (ESP-IDF + FreeRTOS)
 * Target:   ESP32-S3
 *
 * For production flights use ESP-FC (https://github.com/rtlopez/esp-fc).
 * This file is an educational reference skeleton.
 *
 * Build (PlatformIO):
 *   pio run -e esp32s3 -t upload
 *
 * GPIO wiring (matches custom PCB):
 *   GPIO12 → Motor 1 ESC  (Front-Right, CW)
 *   GPIO13 → Motor 2 ESC  (Rear-Left,   CW)
 *   GPIO14 → Motor 3 ESC  (Front-Left,  CCW)
 *   GPIO15 → Motor 4 ESC  (Rear-Right,  CCW)
 *   GPIO16 → Camera tilt servo
 *   GPIO17 → Buzzer (active, 5 V)
 *   GPIO4/5 → I2C SDA/SCL → MPU-6500 IMU + BMP280 barometer
 *   GPIO8/9 → UART1 TX/RX  → GPS (NMEA, 9600 baud)
 *   GPIO6/7 → UART2 TX/RX  → OSD (MSP, 115200 baud)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "driver/ledc.h"
#include "driver/i2c.h"
#include "driver/uart.h"
#include "driver/gpio.h"

#include "esp_log.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */

#define TAG "ESP_BLAST"

/* Motor GPIO pins */
#define PIN_MOTOR_FR  12
#define PIN_MOTOR_RL  13
#define PIN_MOTOR_FL  14
#define PIN_MOTOR_RR  15
#define PIN_TILT_SERVO 16
#define PIN_BUZZER     17

/* I2C */
#define I2C_PORT     I2C_NUM_0
#define I2C_SDA_PIN  4
#define I2C_SCL_PIN  5
#define I2C_CLK_HZ   400000

/* UART1 — GPS */
#define UART_GPS     UART_NUM_1
#define PIN_GPS_TX   8
#define PIN_GPS_RX   9
#define BAUD_GPS     9600

/* UART2 — OSD */
#define UART_OSD     UART_NUM_2
#define PIN_OSD_TX   6
#define PIN_OSD_RX   7
#define BAUD_OSD     115200

/* PWM parameters */
#define PWM_FREQ_HZ  500
#define PWM_MIN_US   1000
#define PWM_MAX_US   2000
#define PWM_MID_US   1500
#define PWM_TIMER_RES LEDC_TIMER_14_BIT   /* 16384 ticks per period */

/* Attitude loop rate */
#define LOOP_HZ      1000

/* RC link-loss failsafe threshold (ms) */
#define RC_TIMEOUT_MS 1000

/* MPU-6500 I2C address & key registers */
#define MPU_ADDR      0x68
#define REG_PWR_MGMT  0x6B
#define REG_GYRO_CFG  0x1B
#define REG_ACCEL_CFG 0x1C
#define REG_ACCEL_OUT 0x3B
#define REG_GYRO_OUT  0x43
#define GYRO_SCALE    131.0f   /* LSB/(°/s) at ±250 °/s */
#define ACCEL_SCALE   16384.0f /* LSB/g at ±2 g */

/* -------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------- */

typedef struct {
    float kp, ki, kd;
    float integral;
    float last_err;
    float i_limit;
} pid_t;

typedef struct {
    float throttle;  /* 0–1000 */
    float roll;      /* -500 to +500 */
    float pitch;
    float yaw;
    int64_t last_rx_us;
} rc_state_t;

typedef enum {
    ARM_DISARMED = 0,
    ARM_ARMING,
    ARM_ARMED,
    ARM_FAILSAFE,
} arm_state_t;

/* -------------------------------------------------------------------------
 * Globals
 * ---------------------------------------------------------------------- */

static rc_state_t   g_rc        = {0};
static arm_state_t  g_arm       = ARM_DISARMED;
static SemaphoreHandle_t g_rc_mutex;

/* LEDC channel → GPIO mapping for motors */
static const int motor_gpios[4] = {
    PIN_MOTOR_FR, PIN_MOTOR_RL, PIN_MOTOR_FL, PIN_MOTOR_RR
};

/* -------------------------------------------------------------------------
 * PWM / LEDC helpers
 * ---------------------------------------------------------------------- */

static uint32_t us_to_duty(uint32_t pulse_us) {
    /* Map pulse_us (1000–2000) to 14-bit LEDC duty for PWM_FREQ_HZ */
    uint32_t period_us = 1000000u / PWM_FREQ_HZ;
    if (pulse_us < PWM_MIN_US) pulse_us = PWM_MIN_US;
    if (pulse_us > PWM_MAX_US) pulse_us = PWM_MAX_US;
    return (uint32_t)(((uint64_t)pulse_us * (1 << 14)) / period_us);
}

static void pwm_set_us(ledc_channel_t channel, uint32_t pulse_us) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, channel, us_to_duty(pulse_us));
    ledc_update_duty(LEDC_LOW_SPEED_MODE, channel);
}

static void pwm_init(void) {
    ledc_timer_config_t timer_cfg = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = PWM_TIMER_RES,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer_cfg);

    for (int i = 0; i < 4; i++) {
        ledc_channel_config_t ch = {
            .gpio_num   = motor_gpios[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel    = (ledc_channel_t)i,
            .timer_sel  = LEDC_TIMER_0,
            .duty       = us_to_duty(PWM_MIN_US),
            .hpoint     = 0,
        };
        ledc_channel_config(&ch);
    }

    /* Camera tilt servo on channel 4 */
    ledc_channel_config_t tilt = {
        .gpio_num   = PIN_TILT_SERVO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_4,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = us_to_duty(PWM_MID_US),
        .hpoint     = 0,
    };
    ledc_channel_config(&tilt);
}

/* -------------------------------------------------------------------------
 * PID controller
 * ---------------------------------------------------------------------- */

static void pid_init(pid_t *pid, float kp, float ki, float kd,
                     float i_limit) {
    pid->kp = kp;  pid->ki = ki;  pid->kd = kd;
    pid->i_limit  = i_limit;
    pid->integral = 0.0f;
    pid->last_err = 0.0f;
}

static float pid_update(pid_t *pid, float setpoint, float measured,
                        float dt) {
    if (dt <= 0.0f) return 0.0f;
    float err = setpoint - measured;
    pid->integral += err * dt;
    if (pid->integral >  pid->i_limit) pid->integral =  pid->i_limit;
    if (pid->integral < -pid->i_limit) pid->integral = -pid->i_limit;
    float deriv = (err - pid->last_err) / dt;
    pid->last_err = err;
    return pid->kp * err + pid->ki * pid->integral + pid->kd * deriv;
}

/* -------------------------------------------------------------------------
 * MPU-6500 I2C driver (minimal)
 * ---------------------------------------------------------------------- */

static esp_err_t mpu_write(uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    return i2c_master_write_to_device(I2C_PORT, MPU_ADDR, buf, 2,
                                      pdMS_TO_TICKS(10));
}

static esp_err_t mpu_read(uint8_t reg, uint8_t *data, size_t len) {
    return i2c_master_write_read_device(I2C_PORT, MPU_ADDR,
                                        &reg, 1, data, len,
                                        pdMS_TO_TICKS(10));
}

static int16_t mpu_word(const uint8_t *buf, int offset) {
    return (int16_t)((buf[offset] << 8) | buf[offset + 1]);
}

static void mpu_init(void) {
    mpu_write(REG_PWR_MGMT,  0x01);  /* clock = gyro X */
    mpu_write(REG_GYRO_CFG,  0x00);  /* ±250 °/s */
    mpu_write(REG_ACCEL_CFG, 0x00);  /* ±2 g */
    vTaskDelay(pdMS_TO_TICKS(50));
    ESP_LOGI(TAG, "MPU-6500 initialised.");
}

static void mpu_read_gyro_dps(float *gx, float *gy, float *gz) {
    uint8_t buf[6];
    mpu_read(REG_GYRO_OUT, buf, 6);
    *gx = mpu_word(buf, 0) / GYRO_SCALE;
    *gy = mpu_word(buf, 2) / GYRO_SCALE;
    *gz = mpu_word(buf, 4) / GYRO_SCALE;
}

static void mpu_read_accel_g(float *ax, float *ay, float *az) {
    uint8_t buf[6];
    mpu_read(REG_ACCEL_OUT, buf, 6);
    *ax = mpu_word(buf, 0) / ACCEL_SCALE;
    *ay = mpu_word(buf, 2) / ACCEL_SCALE;
    *az = mpu_word(buf, 4) / ACCEL_SCALE;
}

/* -------------------------------------------------------------------------
 * Complementary filter for roll / pitch
 * ---------------------------------------------------------------------- */

#define CF_ALPHA 0.98f

static void comp_filter(float *roll, float *pitch,
                        float gx, float gy,
                        float ax, float ay, float az,
                        float dt) {
    float accel_roll  = atan2f(ay, az) * 57.2957795f;
    float accel_pitch = atan2f(-ax, sqrtf(ay*ay + az*az)) * 57.2957795f;
    *roll  = CF_ALPHA * (*roll  + gx * dt) + (1.0f - CF_ALPHA) * accel_roll;
    *pitch = CF_ALPHA * (*pitch + gy * dt) + (1.0f - CF_ALPHA) * accel_pitch;
}

/* -------------------------------------------------------------------------
 * Quad-X motor mixer
 * ---------------------------------------------------------------------- */

static void mix_quad_x(float throttle, float roll, float pitch, float yaw,
                       int32_t out[4]) {
    float m[4];
    m[0] = throttle + roll - pitch - yaw;   /* FR  CW  */
    m[1] = throttle - roll + pitch - yaw;   /* RL  CW  */
    m[2] = throttle + roll + pitch + yaw;   /* FL  CCW */
    m[3] = throttle - roll - pitch + yaw;   /* RR  CCW */
    for (int i = 0; i < 4; i++) {
        int32_t v = (int32_t)m[i];
        if (v <    0) v =    0;
        if (v > 1000) v = 1000;
        out[i] = v;
    }
}

/* -------------------------------------------------------------------------
 * ESP-NOW RC receive callback
 *
 * Packet layout (8 bytes, little-endian):
 *   uint16 throttle  (1000–2000 µs raw)
 *   int16  roll      (-500 to +500)
 *   int16  pitch     (-500 to +500)
 *   int16  yaw       (-500 to +500)
 * ---------------------------------------------------------------------- */

#pragma pack(push, 1)
typedef struct {
    uint16_t throttle_raw;
    int16_t  roll;
    int16_t  pitch;
    int16_t  yaw;
} rc_packet_t;
#pragma pack(pop)

static void espnow_recv_cb(const esp_now_recv_info_t *info,
                           const uint8_t *data, int data_len) {
    if (data_len != (int)sizeof(rc_packet_t)) return;
    const rc_packet_t *pkt = (const rc_packet_t *)data;
    xSemaphoreTake(g_rc_mutex, portMAX_DELAY);
    g_rc.throttle   = (float)(pkt->throttle_raw - 1000);
    /* Scale stick deflection (-500..+500) to physical units:
     *   roll / pitch → target angle  in degrees  (±45 °)
     *   yaw          → target rate   in °/s       (±200 °/s) */
    g_rc.roll       = (float)pkt->roll  * (45.0f  / 500.0f);
    g_rc.pitch      = (float)pkt->pitch * (45.0f  / 500.0f);
    g_rc.yaw        = (float)pkt->yaw   * (200.0f / 500.0f);
    g_rc.last_rx_us = esp_timer_get_time();
    xSemaphoreGive(g_rc_mutex);
}

static void espnow_init(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_start();

    esp_now_init();
    esp_now_register_recv_cb(espnow_recv_cb);
    ESP_LOGI(TAG, "ESP-NOW initialised.");
}

/* -------------------------------------------------------------------------
 * MSP v1 encoder (for OSD telemetry)
 * ---------------------------------------------------------------------- */

static size_t msp_attitude(uint8_t *buf, size_t buf_len,
                           float roll_deg, float pitch_deg,
                           float yaw_deg) {
    /* MSP_ATTITUDE (cmd=108): roll*10, pitch*10, yaw (little-endian) */
    if (buf_len < 9) return 0;
    int16_t r = (int16_t)(roll_deg  * 10.0f);
    int16_t p = (int16_t)(pitch_deg * 10.0f);
    uint16_t y = (uint16_t)fmodf(yaw_deg, 360.0f);

    uint8_t payload[6];
    memcpy(payload + 0, &r, 2);
    memcpy(payload + 2, &p, 2);
    memcpy(payload + 4, &y, 2);

    uint8_t crc = 6 ^ 108;
    for (int i = 0; i < 6; i++) crc ^= payload[i];

    buf[0] = '$'; buf[1] = 'M'; buf[2] = '<';
    buf[3] = 6;   buf[4] = 108;
    memcpy(buf + 5, payload, 6);
    buf[11] = crc;
    return 12;
}

/* -------------------------------------------------------------------------
 * Attitude task (inner loop)
 * ---------------------------------------------------------------------- */

static void attitude_task(void *arg) {
    pid_t pid_roll, pid_pitch, pid_yaw;
    pid_init(&pid_roll,  0.65f, 0.02f, 0.18f, 250.0f);
    pid_init(&pid_pitch, 0.70f, 0.02f, 0.20f, 250.0f);
    pid_init(&pid_yaw,   0.55f, 0.01f, 0.00f, 250.0f);

    float roll_est = 0.0f, pitch_est = 0.0f;
    int64_t t_last = esp_timer_get_time();

    /* Use a separate TickType_t for vTaskDelayUntil */
    TickType_t xLastWake = xTaskGetTickCount();
    const TickType_t period = pdMS_TO_TICKS(1000 / LOOP_HZ);

    for (;;) {
        int64_t now = esp_timer_get_time();
        float dt = (float)(now - t_last) * 1e-6f;
        if (dt < 1e-6f) dt = 1.0f / LOOP_HZ;
        t_last = now;

        float gx, gy, gz, ax, ay, az;
        mpu_read_gyro_dps(&gx, &gy, &gz);
        mpu_read_accel_g(&ax, &ay, &az);

        comp_filter(&roll_est, &pitch_est, gx, gy, ax, ay, az, dt);

        /* Copy RC state thread-safely */
        xSemaphoreTake(g_rc_mutex, portMAX_DELAY);
        rc_state_t rc = g_rc;
        xSemaphoreGive(g_rc_mutex);

        /* RC link-loss check */
        int64_t age_us = esp_timer_get_time() - rc.last_rx_us;
        if (g_arm == ARM_ARMED && age_us > (int64_t)RC_TIMEOUT_MS * 1000) {
            g_arm = ARM_FAILSAFE;
            ESP_LOGW(TAG, "FAILSAFE: RC link lost.");
        }

        if (g_arm == ARM_ARMED) {
            /* roll / pitch: angle mode (setpoint=target °, measured=estimated °)
             * yaw:          rate  mode (setpoint=target °/s, measured=gyro gz °/s) */
            float roll_out  = pid_update(&pid_roll,  rc.roll,  roll_est,  dt);
            float pitch_out = pid_update(&pid_pitch, rc.pitch, pitch_est, dt);
            float yaw_out   = pid_update(&pid_yaw,   rc.yaw,   gz,        dt);

            int32_t motors[4];
            mix_quad_x(rc.throttle, roll_out, pitch_out, yaw_out, motors);

            for (int i = 0; i < 4; i++)
                pwm_set_us((ledc_channel_t)i,
                           (uint32_t)(PWM_MIN_US + motors[i]));
        } else {
            /* Disarmed / failsafe → all motors minimum */
            for (int i = 0; i < 4; i++)
                pwm_set_us((ledc_channel_t)i, PWM_MIN_US);
        }

        vTaskDelayUntil(&xLastWake, period);
    }
}

/* -------------------------------------------------------------------------
 * OSD telemetry task (10 Hz)
 * ---------------------------------------------------------------------- */

static void osd_task(void *arg) {
    uint8_t buf[16];
    for (;;) {
        size_t n = msp_attitude(buf, sizeof(buf), 0.0f, 0.0f, 0.0f);
        uart_write_bytes(UART_OSD, (char *)buf, n);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

/* -------------------------------------------------------------------------
 * Hardware initialisation
 * ---------------------------------------------------------------------- */

static void i2c_init(void) {
    i2c_config_t cfg = {
        .mode             = I2C_MODE_MASTER,
        .sda_io_num       = I2C_SDA_PIN,
        .scl_io_num       = I2C_SCL_PIN,
        .sda_pullup_en    = GPIO_PULLUP_ENABLE,
        .scl_pullup_en    = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_CLK_HZ,
    };
    i2c_param_config(I2C_PORT, &cfg);
    i2c_driver_install(I2C_PORT, I2C_MODE_MASTER, 0, 0, 0);
}

static void uart_init_port(uart_port_t port, int tx, int rx, int baud) {
    uart_config_t cfg = {
        .baud_rate  = baud,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_param_config(port, &cfg);
    uart_set_pin(port, tx, rx, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(port, 256, 0, 0, NULL, 0);
}

static void buzzer_beep(int times) {
    for (int i = 0; i < times; i++) {
        gpio_set_level(PIN_BUZZER, 1);
        vTaskDelay(pdMS_TO_TICKS(150));
        gpio_set_level(PIN_BUZZER, 0);
        vTaskDelay(pdMS_TO_TICKS(150));
    }
}

/* -------------------------------------------------------------------------
 * app_main
 * ---------------------------------------------------------------------- */

void app_main(void) {
    ESP_LOGI(TAG, "ESP-BLAST firmware starting…");

    /* GPIO for buzzer */
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << PIN_BUZZER),
        .mode         = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);
    gpio_set_level(PIN_BUZZER, 0);

    /* Peripheral init */
    i2c_init();
    uart_init_port(UART_GPS, PIN_GPS_TX, PIN_GPS_RX, BAUD_GPS);
    uart_init_port(UART_OSD, PIN_OSD_TX, PIN_OSD_RX, BAUD_OSD);
    pwm_init();
    mpu_init();
    espnow_init();

    g_rc_mutex = xSemaphoreCreateMutex();

    /* ESC arming sequence: hold 1000 µs for 3 s */
    ESP_LOGI(TAG, "ESC arming sequence…");
    buzzer_beep(3);
    vTaskDelay(pdMS_TO_TICKS(2000));
    g_arm = ARM_ARMED;
    ESP_LOGI(TAG, "ARMED.");

    /* Launch tasks */
    xTaskCreatePinnedToCore(attitude_task, "attitude", 4096, NULL,
                            configMAX_PRIORITIES - 1, NULL, 1);
    xTaskCreatePinnedToCore(osd_task,      "osd",      2048, NULL,
                            5,                         NULL, 0);

    ESP_LOGI(TAG, "All tasks launched.");
}
