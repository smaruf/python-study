/*
 * rc_basic.c — Simple RC firmware skeleton (bare-metal C, AVR ATmega328P)
 * Complexity level: SIMPLE
 * Language:         C (no RTOS, no HAL framework)
 *
 * Wiring (Arduino Nano / Uno pinout):
 *   D8  (ICP1)  ← PWM signal from RC receiver (throttle channel)
 *   D9  (OC1A)  → ESC throttle signal
 *   D10 (OC1B)  → Servo 1 (aileron)
 *   D3  (OC2B)  → Servo 2 (elevator)  [separate 8-bit timer]
 *
 * Compile:
 *   avr-gcc -mmcu=atmega328p -DF_CPU=16000000UL -Os -Wall -Wextra \
 *           -o rc_basic.elf rc_basic.c
 *   avr-objcopy -O ihex rc_basic.elf rc_basic.hex
 *   avrdude -c arduino -p m328p -P /dev/ttyUSB0 -b 115200 \
 *           -U flash:w:rc_basic.hex
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <stdint.h>
#include <util/delay.h>

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */
#define F_CPU_MHZ       16UL
/* Timer1 prescaler 8 → tick = 0.5 µs, so 1 µs = 2 ticks */
#define US_TO_TICKS(us) ((us) * 2U)

#define PWM_MIN_TICKS   US_TO_TICKS(1000)
#define PWM_MAX_TICKS   US_TO_TICKS(2000)
#define PWM_MID_TICKS   US_TO_TICKS(1500)

/* --------------------------------------------------------------------------
 * Global (ISR-shared) variables  — always volatile
 * -------------------------------------------------------------------------- */
static volatile uint16_t g_pwm_in_ticks = PWM_MID_TICKS;  /* measured input */
static volatile uint16_t g_rise_time    = 0;
static volatile uint8_t  g_new_sample   = 0;               /* flag: new value ready */

/* --------------------------------------------------------------------------
 * Timer1 Input Capture — measures incoming PWM pulse width
 * -------------------------------------------------------------------------- */
ISR(TIMER1_CAPT_vect) {
    if (TCCR1B & (1 << ICES1)) {
        /* Rising edge captured: save timestamp */
        g_rise_time = ICR1;
        TCCR1B &= ~(1 << ICES1);   /* switch to falling-edge capture */
    } else {
        /* Falling edge: compute pulse width */
        uint16_t width = ICR1 - g_rise_time;
        /* Clamp to valid servo range */
        if (width >= PWM_MIN_TICKS && width <= PWM_MAX_TICKS) {
            g_pwm_in_ticks = width;
            g_new_sample   = 1;
        }
        TCCR1B |= (1 << ICES1);    /* switch back to rising-edge capture */
    }
}

/* --------------------------------------------------------------------------
 * Timer1 PWM output (OC1A = D9, OC1B = D10)
 * Fast PWM, mode 14 (ICR1 = TOP), prescaler 8
 * -------------------------------------------------------------------------- */
static void timer1_init(void) {
    /* Set TOP to 20 ms period (20000 µs × 2 ticks/µs = 40000) */
    ICR1  = 40000U;
    OCR1A = PWM_MID_TICKS;   /* ESC: arm at 1500 µs */
    OCR1B = PWM_MID_TICKS;   /* Servo aileron: centred */

    /* Mode 14: Fast PWM, TOP=ICR1 */
    TCCR1A = (1 << COM1A1) | (1 << COM1B1) | (1 << WGM11);
    TCCR1B = (1 << WGM13)  | (1 << WGM12)
           | (1 << ICES1)                   /* ICP: start on rising edge */
           | (1 << CS11);                   /* prescaler 8 */

    TIMSK1 = (1 << ICIE1);  /* enable input capture interrupt */

    /* OC1A (D9) and OC1B (D10) as outputs */
    DDRB |= (1 << PB1) | (1 << PB2);
}

/* --------------------------------------------------------------------------
 * Clamp helper
 * -------------------------------------------------------------------------- */
static inline uint16_t clamp16(uint16_t val, uint16_t lo, uint16_t hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main(void) {
    timer1_init();
    sei();  /* enable global interrupts */

    /* ESC arming: hold 1000 µs for ~2 seconds */
    OCR1A = PWM_MIN_TICKS;
    _delay_ms(2000);

    while (1) {
        if (g_new_sample) {
            g_new_sample = 0;

            uint16_t pulse = g_pwm_in_ticks;
            pulse = clamp16(pulse, PWM_MIN_TICKS, PWM_MAX_TICKS);

            /* Pass-through: input → ESC and servo (no stabilisation) */
            OCR1A = pulse;   /* ESC throttle */
            OCR1B = pulse;   /* Servo aileron (demo: mirrors throttle) */
        }
        /* 50 Hz rate: wait ~20 ms before next check */
        _delay_ms(20);
    }
    return 0;  /* never reached */
}
