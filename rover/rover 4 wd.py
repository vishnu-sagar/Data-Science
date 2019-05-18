import time
import serial
import RPi.GPIO as GPIO
TRIG = 5
ECHO = 6
led=13
led2=24
led1=26
m11=17
m12=27
m21=22
m22=23
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG,GPIO.OUT)                  # initialize GPIO Pin as outputs
GPIO.setup(ECHO,GPIO.IN)                   # initialize GPIO Pin as input
GPIO.setup(led,GPIO.OUT)                  
GPIO.setup(m11,GPIO.OUT)
GPIO.setup(m12,GPIO.OUT)
GPIO.setup(m21,GPIO.OUT)
GPIO.setup(m22,GPIO.OUT)
GPIO.setup(led2,GPIO.OUT)
GPIO.setup(led1,GPIO.OUT)
GPIO.output(led, 1)
time.sleep(5)
GPIO.setwarnings(False)
time.sleep(5)
ser=serial.Serial(port='/dev/ttyS0',baudrate=9600,timeout=1)
time.sleep(1)
def stop():
    print "stop"
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)
def forward():
    GPIO.output(m11, 1)
    GPIO.output(m12, 1)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)
    print "Forward"
def back():
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 1)
    print "back"
def left():
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 1)
    print "left"
def right():
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)
    GPIO.output(m11, 0)
    GPIO.output(m12, 1)
    print "right"

def manual():
  while True:
    s=ser.read()
    
    if s=="5":
     GPIO.output(led1,1)
     print "auto"
     auto()
    if s == "1":
     print s
     forward()
    if s == "2":
     print s
     back()
     
    if s == "3":
     print s
     left()
     print "left"
    if s == "4":
     print s
     right()
     
    if s == "0":
     print s
     
     stop()
      
   
def auto():
 stop()
 count=0
 while True:
  s=ser.read()
  if s=="6":
   GPIO.output(led1,0)
   GPIO.output(led2,1)
   print "manual"
   manual()
   
  i=0
  avgDistance=0
  for i in range(5):
   GPIO.output(TRIG, False)                 #Set TRIG as LOW
   time.sleep(0.1)                                   #Delay
   GPIO.output(TRIG, True)                  #Set TRIG as HIGH
   time.sleep(0.00001)                           #Delay of 0.00001 seconds
   GPIO.output(TRIG, False)                 #Set TRIG as LOW
   while GPIO.input(ECHO)==0:              #Check whether the ECHO is LOW
       GPIO.output(led, False)             
   pulse_start = time.time()
   while GPIO.input(ECHO)==1:              #Check whether the ECHO is HIGH
       GPIO.output(led, False) 
   pulse_end = time.time()
   pulse_duration = pulse_end - pulse_start #time to get back the pulse to sensor
   distance = pulse_duration * 17150        #Multiply pulse duration by 17150 (34300/2) to get distance
   distance = round(distance,2)                 #Round to two decimal points
   avgDistance=avgDistance+distance
  avgDistance=avgDistance/5
  print avgDistance
  flag=0
  if s=="6":
   GPIO.output(led1,0)
   GPIO.output(led2,1)
   manual()
   print "manual"
  if avgDistance < 15:      #Check whether the distance is within 15 cm range
    count=count+1
    stop()
    time.sleep(1)
    back()
    time.sleep(1.5)
    if (count%3 ==1) & (flag==0):
     right()
     flag=1
    else:
     left()
     flag=0
    time.sleep(1.5)
    stop()
    time.sleep(1)
  else:
    forward()
    flag=0
  
     
while True:
 GPIO.output(led1,1)
 auto()
    


