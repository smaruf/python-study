import touchio
import time
from board import *
from digitalio import DigitalInOut, Direction
led = DigitalInOut(GP15)
led.direction = Direction.OUTPUT
led_state = 0
touch_pin = touchio.TouchIn(GP16)
while True:
   print("The pin state is %s" % touch_pin.value)
   if touch_pin.value == True and led_state == False:
       led.value = True
       led_state = 1
       time.sleep(0.5)
   elif touch_pin.value == True and led_state == True:
       led.value = False
       led_state = 0
       time.sleep(0.5)
time.sleep(0.1)
#'''https://www.tomshardware.com/how-to/raspberry-pi-pico-banana-touch-input'''
