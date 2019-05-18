import time
import serial
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)  
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)


GPIO.setwarnings(False)
time.sleep(5)

ser=serial.Serial(port='/dev/ttyS0',baudrate=9600,timeout=1)
time.sleep(1)
while True:
    s=ser.read()
    print s
    if s == "1":
     GPIO.output(17,1)
     GPIO.output(m12, 0)
     GPIO.output(m21, 0)
     GPIO.output(m22, 0)
     print "Forward"
    if s == "2":
     GPIO.output(17, 0)
     GPIO.output(27, 1)
     GPIO.output(22, 0)
     GPIO.output(23, 0)
     print "back"
    if s == "3":
     GPIO.output(17, 1)
     GPIO.output(27, 0)
     GPIO.output(22, 1)
     GPIO.output(23, 0)
     print "left"
    if s == "4":
     GPIO.output(17, 1)
     GPIO.output(27, 0)
     GPIO.output(22, 0)
     GPIO.output(23, 1)
     print "right"
    if s == "0":
     print "stop"
     GPIO.output(17, 0)
     GPIO.output(27, 0)
     GPIO.output(22, 0)
     GPIO.output(23, 0)
     



        
    
    
     
