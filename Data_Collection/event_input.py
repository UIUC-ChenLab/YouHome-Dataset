import argparse
import os
import picamera
import RPi.GPIO as gpio
import time
import datetime
import pytz
import numpy as np
import pandas as pd
import board
import adafruit_dht
import adafruit_tsl2591
import serial
import busio


def getTempHum():
    dht22 = adafruit_dht.DHT22(board.D14, use_pulseio=False)
    try:
        temperature, humidity = dht22.temperature, dht22.humidity
    except RuntimeError:
        temperature, humidity = -1, -1
    output = np.array([temperature, humidity])
    return output

def getDistance():
    uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
    us100 = adafruit_us100.US100(uart)
    try:
        distance = us100.distance
    except RuntimeError:
        distance = -1
    return np.array([us100.distance])

def getMotion():
    gpio.setmode(gpio.BCM)
    PIR_PIN = 23
    gpio.setup(PIR_PIN, gpio.IN)
    if (gpio.input(PIR_PIN) == 1):
        detected = 1
    else:
        detected = 0
    return np.array([detected])


def getLight():
    # Initialize the I2C bus.
    i2c = busio.I2C(board.SCL, board.SDA)
    # Initialize the sensor.
    sensor = adafruit_tsl2591.TSL2591(i2c)
    lux = sensor.lux
    infrared = sensor.infrared
    visible = sensor.visible
    full_spectrum = sensor.full_spectrum
    output = np.array([lux, infrared, visible, full_spectrum])
    return np.array([lux, infrared, visible, full_spectrum])

def eventInput(timeInt, path, name):
    tz = pytz.timezone('America/Chicago')
    timeInterval = int(timeInt)
    eventOut = np.zeros(shape=(timeInterval,7))
    eventTitle = ["Temperature", "Humidity", "lux", "infrared", "visible", "full_spectrum", "motion"]
    eventTime = []
    startTime = datetime.datetime.now(tz).strftime('%Y.%m.%d %H:%M:%S')
    for i in range(timeInterval):
        currentTime = datetime.datetime.now(tz).strftime('%Y.%m.%d %H:%M:%S')
        eventTime.append(currentTime)
        temphumTemp = getTempHum()
        light = getLight()
        motion = getMotion()
        eventOut[i, :] = np.append(np.append(temphumTemp, light), motion)
        time.sleep(0.65)
    endTime = datetime.datetime.now(tz).strftime('%Y.%m.%d %H:%M:%S')
    dfData = pd.DataFrame(columns=eventTitle, index=eventTime, data=eventOut)
    dfData.to_csv(path+name+'.csv', encoding='gbk')
    print("event start time: is " +  startTime)
    print("event end time: is " +  endTime)
    return

parser = argparse.ArgumentParser(description='Input for User, Event Name')
parser.add_argument('--user', '-u', help='name of the user')
parser.add_argument('--event', '-e', help='name of the event', required=True)
parser.add_argument('--number', '-n', help='number of the event', required=True)
parser.add_argument('--time', '-t', help='time interval(sec)', default = 10)
parser.add_argument('--camera', '-c', help='node number', required=True)
args = parser.parse_args()

if __name__ == '__main__':
    path = "./" + args.user + "/" + args.event + args.number + "/"
    name = "00" + args.user + "_c" + args.camera + "s" + args.number
    eventInput(args.time, path, name)