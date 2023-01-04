import picamera
import argparse
import time
import pytz
import datetime
import os

def save_video(path, timeinterval, name):
    recordtime = int(timeinterval)
    tz = pytz.timezone('America/Chicago')
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        startTime = datetime.datetime.now(tz).strftime('%Y.%m.%d %H:%M:%S')
        camera.start_recording(path + name+'.h264')
        camera.wait_recording(recordtime)
        camera.stop_recording()
    endTime = datetime.datetime.now(tz).strftime('%Y.%m.%d %H:%M:%S')
    camera.close()
    print("camera start time: is " +  startTime)
    print("camera end time: is " +  endTime)

def save_img(path, timeinterval):
    video_path = path
    videos = os.listdir(video_path)
    file_name = timeinterval + '.h264'
    folder_name = timeinterval
    os.makedirs(path + folder_name,exist_ok=True)
    vc = cv2.VideoCapture(path + file_name) #load the video
    c=0
    rval=vc.isOpened()
    while rval:   #loop to read the frame
        c = c + 1
        rval, frame = vc.read()
        pic_path = path + folder_name +'/'
        if rval:
            cv2.imwrite(pic_path + str(c) + '.jpg', frame) #store as file+number.jpg
            cv2.waitKey(1)
        else:
            break
    vc.release()
    # print('save_success')

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
    save_video(path, args.time, name)
    # save_img(path, args.time)
