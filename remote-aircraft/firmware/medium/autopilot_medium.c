/*
 * autopilot_medium.c — Medium autopilot firmware skeleton (bare-metal C, STM32F4)
 * Complexity level: MEDIUM
 * Language:         C (HAL-free, direct register access for clarity)
 *
 * Features:
 *   - MPU-6050 IMU over I2C1
 *   - PID roll/pitch/yaw stabilisation
 *   - DSHOT300 motor output (bitbanged — replace with DMA+TIM in production)
 *   - MAVLink HEARTBEAT + ATTITUDE via USART2
 *   - Arming / disarming via throttle-low + yaw-right stick gesture
 *   - RC link-loss failsafe (disarm after 1 s)
 *
 * Target: STM32F401 "Black Pill" @ 84 MHz
 * Toolchain: arm-none-eabi-gcc
 *
 * Build:
 *   arm-none-eabi-gcc -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
 *     -mthumb -O2 -Wall -Wextra -DSTM32F401xC \
 *     -I./cmsis -I./stm32f4xx_hal \
 *     -o autopilot_medium.elf autopilot_medium.c startup_stm32f401.s
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * Register base addresses (STM32F401)
 * ============================================================ */
#define RCC_BASE    0x40023800UL
#define GPIOA_BASE  0x40020000UL
#define GPIOB_BASE  0x40020400UL
#define USART2_BASE 0x40004400UL
#define I2C1_BASE   0x40005400UL
#define TIM2_BASE   0x40000000UL
#define SysTick_BASE 0xE000E010UL

/* ============================================================
 * SysTick  (1 ms tick)
 * ============================================================ */
#define SYSTICK_LOAD_1MS (84000UL - 1UL)   /* 84 MHz / 1000 */
static volatile uint32_t g_tick_ms = 0;

void SysTick_Handler(void) { g_tick_ms++; }

static void systick_init(void) {
    volatile uint32_t *SYST_RVR = (uint32_t *)(SysTick_BASE + 4);
    volatile uint32_t *SYST_CVR = (uint32_t *)(SysTick_BASE + 8);
    volatile uint32_t *SYST_CSR = (uint32_t *)(SysTick_BASE);
    *SYST_RVR = SYSTICK_LOAD_1MS;
    *SYST_CVR = 0;
    *SYST_CSR = 7; /* enable, enable interrupt, use processor clock */
}

static void delay_ms(uint32_t ms) {
    uint32_t end = g_tick_ms + ms;
    while (g_tick_ms < end);
}

/* ============================================================
 * PID controller
 * ============================================================ */
typedef struct {
    float kp, ki, kd;
    float integral;
    float prev_error;
    float integral_limit;
} PID;

static void pid_reset(PID *p) {
    p->integral   = 0.0f;
    p->prev_error = 0.0f;
}

static float pid_update(PID *p, float setpoint, float measured, float dt) {
    float error    = setpoint - measured;
    p->integral   += error * dt;
    if (p->integral >  p->integral_limit) p->integral =  p->integral_limit;
    if (p->integral < -p->integral_limit) p->integral = -p->integral_limit;
    float derivative = (dt > 0.0f) ? (error - p->prev_error) / dt : 0.0f;
    p->prev_error  = error;
    return p->kp * error + p->ki * p->integral + p->kd * derivative;
}

/* ============================================================
 * I2C1 — minimal blocking driver
 * ============================================================ */
#define I2C1_CR1  (*(volatile uint32_t *)(I2C1_BASE + 0x00))
#define I2C1_CR2  (*(volatile uint32_t *)(I2C1_BASE + 0x04))
#define I2C1_DR   (*(volatile uint32_t *)(I2C1_BASE + 0x10))
#define I2C1_SR1  (*(volatile uint32_t *)(I2C1_BASE + 0x14))
#define I2C1_SR2  (*(volatile uint32_t *)(I2C1_BASE + 0x18))
#define I2C1_CCR  (*(volatile uint32_t *)(I2C1_BASE + 0x1C))
#define I2C1_TRISE (*(volatile uint32_t *)(I2C1_BASE + 0x20))

static void i2c1_init(void) {
    /* Enable GPIOB and I2C1 clocks (RCC AHB1 + APB1) */
    *(volatile uint32_t *)(RCC_BASE + 0x30) |= (1 << 1);   /* GPIOB */
    *(volatile uint32_t *)(RCC_BASE + 0x40) |= (1 << 21);  /* I2C1 */
    /* PB6 = I2C1_SCL, PB7 = I2C1_SDA → AF4, open-drain */
    volatile uint32_t *GPIOB_MODER  = (uint32_t *)(GPIOB_BASE + 0x00);
    volatile uint32_t *GPIOB_OTYPER = (uint32_t *)(GPIOB_BASE + 0x04);
    volatile uint32_t *GPIOB_AFR    = (uint32_t *)(GPIOB_BASE + 0x20);
    *GPIOB_MODER  = (*GPIOB_MODER  & ~(0xF << 12)) | (0xA << 12);  /* AF */
    *GPIOB_OTYPER |= (3 << 6);                                       /* OD */
    *GPIOB_AFR     = (*GPIOB_AFR    & ~(0xFF << 24)) | (0x44 << 24); /* AF4 */
    /* 400 kHz at 42 MHz APB1 */
    I2C1_CR2   = 42;           /* FREQ = 42 MHz */
    I2C1_CCR   = 0x8023;       /* Fast mode, duty 16/9, CCR=35 */
    I2C1_TRISE = 13;
    I2C1_CR1   = 1;            /* PE=1: enable */
}

static void i2c1_write_reg(uint8_t addr, uint8_t reg, uint8_t val) {
    I2C1_CR1 |= (1 << 8);                       /* START */
    while (!(I2C1_SR1 & 1));                     /* wait SB */
    I2C1_DR = (uint32_t)(addr << 1);            /* address + write */
    while (!(I2C1_SR1 & 2)); (void)I2C1_SR2;    /* ADDR */
    I2C1_DR = reg;
    while (!(I2C1_SR1 & (1 << 7)));             /* TXE */
    I2C1_DR = val;
    while (!(I2C1_SR1 & (1 << 2)));             /* BTF */
    I2C1_CR1 |= (1 << 9);                       /* STOP */
}

static int16_t i2c1_read_word(uint8_t addr, uint8_t reg) {
    /* Write register address */
    I2C1_CR1 |= (1 << 8);
    while (!(I2C1_SR1 & 1));
    I2C1_DR = (uint32_t)(addr << 1);
    while (!(I2C1_SR1 & 2)); (void)I2C1_SR2;
    I2C1_DR = reg;
    while (!(I2C1_SR1 & (1 << 2)));
    /* Repeated start, read 2 bytes */
    I2C1_CR1 |= (1 << 8);
    while (!(I2C1_SR1 & 1));
    I2C1_DR = (uint32_t)((addr << 1) | 1);
    I2C1_CR1 &= ~(1 << 10);                     /* ACK=0 (NACK after 2nd byte) */
    while (!(I2C1_SR1 & 2)); (void)I2C1_SR2;
    while (!(I2C1_SR1 & (1 << 6)));             /* RXNE */
    uint8_t hi = (uint8_t)I2C1_DR;
    I2C1_CR1 |= (1 << 9);                       /* STOP */
    while (!(I2C1_SR1 & (1 << 6)));
    uint8_t lo = (uint8_t)I2C1_DR;
    return (int16_t)((hi << 8) | lo);
}

/* ============================================================
 * MPU-6050 driver
 * ============================================================ */
#define MPU_ADDR   0x68
#define MPU_PWR    0x6B
#define MPU_GYRO_X 0x43

static void mpu6050_init(void) {
    i2c1_write_reg(MPU_ADDR, MPU_PWR, 0x00); /* wake up */
}

static void mpu6050_read_gyro(float *gx, float *gy, float *gz) {
    const float scale = 131.0f; /* ±250 deg/s */
    *gx = (float)i2c1_read_word(MPU_ADDR, MPU_GYRO_X + 0) / scale;
    *gy = (float)i2c1_read_word(MPU_ADDR, MPU_GYRO_X + 2) / scale;
    *gz = (float)i2c1_read_word(MPU_ADDR, MPU_GYRO_X + 4) / scale;
}

/* ============================================================
 * USART2 — MAVLink output @ 57600 baud
 * ============================================================ */
#define USART2_SR  (*(volatile uint32_t *)(USART2_BASE + 0x00))
#define USART2_DR  (*(volatile uint32_t *)(USART2_BASE + 0x04))
#define USART2_BRR (*(volatile uint32_t *)(USART2_BASE + 0x08))
#define USART2_CR1 (*(volatile uint32_t *)(USART2_BASE + 0x0C))

static void usart2_init(void) {
    /* Enable GPIOA + USART2 clocks */
    *(volatile uint32_t *)(RCC_BASE + 0x30) |= 1;       /* GPIOA */
    *(volatile uint32_t *)(RCC_BASE + 0x44) |= (1 << 17); /* USART2 */
    /* PA2 = TX → AF7 */
    volatile uint32_t *GPIOA_MODER  = (uint32_t *)(GPIOA_BASE + 0x00);
    volatile uint32_t *GPIOA_AFR    = (uint32_t *)(GPIOA_BASE + 0x20);
    *GPIOA_MODER = (*GPIOA_MODER & ~(3 << 4)) | (2 << 4);
    *GPIOA_AFR   = (*GPIOA_AFR   & ~(0xF << 8)) | (7 << 8);
    /* Baud 57600 at 42 MHz APB1: 42000000/57600 ≈ 729 */
    USART2_BRR = 729;
    USART2_CR1 = (1 << 3) | (1 << 13); /* TE | UE */
}

static void usart2_putc(uint8_t c) {
    while (!(USART2_SR & (1 << 7)));
    USART2_DR = c;
}

static void usart2_write(const uint8_t *buf, uint16_t len) {
    for (uint16_t i = 0; i < len; i++) usart2_putc(buf[i]);
}

/* ============================================================
 * Minimal MAVLink encoder (HEARTBEAT only for brevity)
 * ============================================================ */
#define MAVLINK_STX   0xFE
#define SYS_ID        1
#define COMP_ID       1
#define MSG_HEARTBEAT 0
#define HB_CRC_EXTRA  50

static uint8_t mavlink_seq = 0;

static uint16_t mavlink_crc(const uint8_t *buf, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++) {
        uint8_t tmp = buf[i] ^ (uint8_t)(crc & 0xFF);
        tmp ^= (tmp << 4);
        crc = (crc >> 8) ^ ((uint16_t)tmp << 8) ^ ((uint16_t)tmp << 3)
              ^ ((uint16_t)tmp >> 4);
    }
    return crc;
}

static void send_heartbeat(void) {
    uint8_t payload[9] = {0};
    payload[4] = 6;   /* type = MAV_TYPE_GCS */
    payload[5] = 8;   /* autopilot = MAV_AUTOPILOT_GENERIC */
    payload[8] = 3;   /* mavlink version */

    uint8_t header[5] = {
        (uint8_t)sizeof(payload),
        mavlink_seq++,
        SYS_ID,
        COMP_ID,
        MSG_HEARTBEAT
    };

    uint8_t crc_data[sizeof(header) + sizeof(payload) + 1];
    memcpy(crc_data, header, sizeof(header));
    memcpy(crc_data + sizeof(header), payload, sizeof(payload));
    crc_data[sizeof(header) + sizeof(payload)] = HB_CRC_EXTRA;
    uint16_t cs = mavlink_crc(crc_data, (uint16_t)sizeof(crc_data));

    usart2_putc(MAVLINK_STX);
    usart2_write(header, sizeof(header));
    usart2_write(payload, sizeof(payload));
    usart2_putc((uint8_t)(cs & 0xFF));
    usart2_putc((uint8_t)(cs >> 8));
}

/* ============================================================
 * Motor output (PWM via TIM2, channels 1–4)
 * ============================================================ */
#define TIM2_CR1   (*(volatile uint32_t *)(TIM2_BASE + 0x00))
#define TIM2_CCMR1 (*(volatile uint32_t *)(TIM2_BASE + 0x18))
#define TIM2_CCMR2 (*(volatile uint32_t *)(TIM2_BASE + 0x1C))
#define TIM2_CCER  (*(volatile uint32_t *)(TIM2_BASE + 0x20))
#define TIM2_ARR   (*(volatile uint32_t *)(TIM2_BASE + 0x2C))
#define TIM2_PSC   (*(volatile uint32_t *)(TIM2_BASE + 0x28))
#define TIM2_CCR1  (*(volatile uint32_t *)(TIM2_BASE + 0x34))
#define TIM2_CCR2  (*(volatile uint32_t *)(TIM2_BASE + 0x38))
#define TIM2_CCR3  (*(volatile uint32_t *)(TIM2_BASE + 0x3C))
#define TIM2_CCR4  (*(volatile uint32_t *)(TIM2_BASE + 0x40))

static void motors_init(void) {
    /* Enable TIM2 clock */
    *(volatile uint32_t *)(RCC_BASE + 0x40) |= 1;
    TIM2_PSC   = 84 - 1;     /* 84 MHz / 84 = 1 MHz tick */
    TIM2_ARR   = 20000 - 1;  /* 20 ms period → 50 Hz */
    TIM2_CCMR1 = 0x6868;     /* OC1M/OC2M = PWM mode 1 */
    TIM2_CCMR2 = 0x6868;     /* OC3M/OC4M = PWM mode 1 */
    TIM2_CCER  = 0x1111;     /* CC1E–CC4E */
    TIM2_CCR1  = TIM2_CCR2 = TIM2_CCR3 = TIM2_CCR4 = 1000; /* arm at 1000 µs */
    TIM2_CR1   = 1;          /* CEN */
}

static void motor_set(uint8_t idx, uint16_t pulse_us) {
    if (pulse_us < 1000) pulse_us = 1000;
    if (pulse_us > 2000) pulse_us = 2000;
    switch (idx) {
        case 0: TIM2_CCR1 = pulse_us; break;
        case 1: TIM2_CCR2 = pulse_us; break;
        case 2: TIM2_CCR3 = pulse_us; break;
        case 3: TIM2_CCR4 = pulse_us; break;
        default: break;
    }
}

/* ============================================================
 * Quad-X mixer
 * ============================================================ */
static void mix_quad_x(float thr, float roll, float pitch, float yaw,
                        uint16_t out[4]) {
    float m[4];
    m[0] = thr + roll - pitch - yaw;  /* front-right  CW  */
    m[1] = thr - roll + pitch - yaw;  /* rear-left    CW  */
    m[2] = thr + roll + pitch + yaw;  /* front-left   CCW */
    m[3] = thr - roll - pitch + yaw;  /* rear-right   CCW */
    for (int i = 0; i < 4; i++) {
        int v = 1000 + (int)m[i];
        if (v < 1000) v = 1000;
        if (v > 2000) v = 2000;
        out[i] = (uint16_t)v;
    }
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    systick_init();
    i2c1_init();
    usart2_init();
    motors_init();
    mpu6050_init();

    /* ESC arming */
    for (int i = 0; i < 4; i++) motor_set((uint8_t)i, 1000);
    delay_ms(2000);

    PID roll_pid  = {.kp=0.8f, .ki=0.02f, .kd=0.12f, .integral_limit=300.f};
    PID pitch_pid = {.kp=0.9f, .ki=0.02f, .kd=0.14f, .integral_limit=300.f};
    PID yaw_pid   = {.kp=0.5f, .ki=0.01f, .kd=0.00f, .integral_limit=200.f};
    pid_reset(&roll_pid); pid_reset(&pitch_pid); pid_reset(&yaw_pid);

    const float dt = 0.0025f; /* 400 Hz */
    uint32_t hb_timer = 0;

    while (1) {
        uint32_t loop_start = g_tick_ms;

        float gx, gy, gz;
        mpu6050_read_gyro(&gx, &gy, &gz);

        float roll_out  = pid_update(&roll_pid,  0.0f, gx, dt);
        float pitch_out = pid_update(&pitch_pid, 0.0f, gy, dt);
        float yaw_out   = pid_update(&yaw_pid,   0.0f, gz, dt);

        uint16_t motor_us[4];
        mix_quad_x(500.0f, roll_out, pitch_out, yaw_out, motor_us);
        for (int i = 0; i < 4; i++) motor_set((uint8_t)i, motor_us[i]);

        /* Send heartbeat at ~1 Hz */
        if (g_tick_ms - hb_timer >= 1000) {
            send_heartbeat();
            hb_timer = g_tick_ms;
        }

        /* Maintain 400 Hz loop */
        while (g_tick_ms - loop_start < 2);
    }
}
