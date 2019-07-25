######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import csv
import datetime


def csv_write(file_name, rights, counter, data):
    with open(file_name, rights, newline='') as f:
        fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
        thewriter = csv.DictWriter(f, fieldnames = fields)
        counter 
        thewriter.writerow(data)
    return;


def erase_csv(file_name):
    f = open(file_name, "w")
    f.truncate()
    f.close()


##MICROSECONDS = currentDT.microsecond


#All the logs to bve initially stored
with open('person_log.csv', 'w', newline='') as f:
    fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
    thewriter = csv.DictWriter(f, fieldnames = fields)
    thewriter.writeheader()

#Minute averaged data to be stored here
with open('minute_logs.csv', 'w', newline='') as f:
    fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
    thewriter = csv.DictWriter(f, fieldnames = fields)
    thewriter.writeheader()
    
with open('hour_logs.csv', 'w', newline='') as f:
    fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
    thewriter = csv.DictWriter(f, fieldnames = fields)
    thewriter.writeheader()

#All days data to be represented to be shown here.
with open('day_logs.csv', 'w', newline='') as f:
    fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
    thewriter = csv.DictWriter(f, fieldnames = fields)
    thewriter.writeheader()

#All weeks data to be logged here
with open('week_logs.csv', 'w', newline='') as f:
    fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
    thewriter = csv.DictWriter(f, fieldnames = fields)
    thewriter.writeheader()



csv_counter = 0
second_counter = 0
minute_counter = 0
hour_counter = 0
day_counter = 0
month_counter = 0
HOUR_FLAG = 0
MINUTE_FLAG = 0
DAY_FLAG = 0
MONTH_FLAG = 0
    
# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 2
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 20
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        number_of_objects = int(num[0])
        classes_without_person = classes[np.nonzero(classes>1.00)]
        number_of_classes = len(classes_without_person)
        number_of_persons = number_of_objects - number_of_classes
        #print("Original: ")
        #print(classes)
        #print('\n\r')
        print("Number of People in the Room: ")
        print(number_of_persons)
        print('\n\r')
         
        currentDT = datetime.datetime.now()    #updating the current time
        
        #fetching all the Time values
        YEAR = currentDT.year
        MONTH = currentDT.month
        DATE = currentDT.day
        HOUR = currentDT.hour
        MINUTES = currentDT.minute
        SECONDS = currentDT.second
        
        #setting flags as to decide when to start or stop updates
        
        if(MINUTE_FLAG == 0):
            minute_temp = int(MINUTES)
            print("into_minute_flg")
            MINUTE_FLAG = 1
            second_counter = 0
            
        if(HOUR_FLAG == 0):
            hour_temp = int(HOUR)
            print("into_hour_flg")
            HOUR_FLAG = 1
            
        if(DAY_FLAG == 0):
            day_temp = int(DATE)
            print("into_daye_flg")
            DAY_FLAG = 1
            
        # putting everything in the person logger . It can store data upto 6 hours
        with open('person_log.csv', 'a', newline='') as f:
            fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
            thewriter = csv.DictWriter(f, fieldnames = fields)
            csv_counter = csv_counter + 1 
            thewriter.writerow({'YEAR': str(YEAR), 'MONTH': str(MONTH), 'DATE': str(DATE), 'HOUR': str(HOUR), 'MINUTES': str(MINUTES), 'SECONDS':str(SECONDS), 'NUMBER':str(number_of_persons)})
        
        if(csv_counter >= 21600):     #entries upto six hours is only stored.
            f = open("person_log.csv", "w")
            f.truncate()
            f.close()
            csv_counter = 0
            print(csv_counter)          

        
        if((minute_temp - MINUTES) == 0):
            #updating the csv
            with open('minute_logs.csv', 'a', newline='') as f:
                fields = ['YEAR', 'MONTH', 'DATE', 'HOUR', 'MINUTES', 'SECONDS', 'NUMBER']
                thewriter = csv.DictWriter(f, fieldnames = fields)
                second_counter = second_counter + 1 
                thewriter.writerow({'YEAR': str(YEAR), 'MONTH': str(MONTH), 'DATE': str(DATE), 'HOUR': str(HOUR), 'MINUTES': str(MINUTES), 'SECONDS':str(SECONDS), 'NUMBER':str(number_of_persons)})
        else:
            MINUTE_FLAG = 0
            with open("minute_logs.csv", "r") as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                del data[0]
                rows = len(data)  #this is the way you decide the number of rows in a list
                total_entries = rows
                columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
                add = 0
                for i in range(rows):
                    print("The sum before add is :", add)
                    add = add + int(data[i][6])
                    if((int(data[i][6])) == 0):
                        total_entries = total_entries - 1
                #del data[0]
                print("The final addition is:", add)
                print("The total number of entries are :", total_entries)
                if(total_entries):
                    minutes_average = add/total_entries
                else:
                    minutes_average = 0
                minutes_average = int(minutes_average)
            csv_write('hour_logs.csv', 'a',hour_counter, {'YEAR': str(YEAR), 'MONTH': str(MONTH), 'DATE': str(DATE), 'HOUR': str(HOUR), 'MINUTES': str(MINUTES), 'SECONDS':str(SECONDS), 'NUMBER':str(minutes_average)})
            erase_csv("minute_logs.csv")
            
            
        
        if((hour_temp - HOUR) != 0):
            HOUR_FLAG = 0
            with open("hour_logs.csv", "r") as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                del data[0]
                rows = len(data)  #this is the way you decide the number of rows in a list
                total_entries = rows
                columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
                add = 0
                for i in range(rows):
                    print("The sum before add is :", add)
                    add = add + int(data[i][6])
                    if((int(data[i][6])) == 0):
                        total_entries = total_entries - 1
                #del data[0]
                print("The final addition is:", add)
                print("The total number of entries are :", total_entries)
                if(total_entries):
                    hour_average = add/total_entries
                else:
                    hour_average = 0
                hour_average = int(hour_average) 
            csv_write('day_logs.csv', 'a',hour_counter, {'YEAR': str(YEAR), 'MONTH': str(MONTH), 'DATE': str(DATE), 'HOUR': str(HOUR), 'MINUTES': str(MINUTES), 'SECONDS':str(SECONDS), 'NUMBER':str(hour_average)})
        

        if((day_temp - DATE) != 0):
            DAY_FLAG = 0
            with open("hour_logs.csv", "r") as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                del data[0]
                rows = len(data)  #this is the way you decide the number of rows in a list
                total_entries = rows
                columns = len(data[0]) # this is the way to determine the nmumber of columns in the list
                add = 0
                for i in range(rows):
                    print("The sum before add is :", add)
                    add = add + int(data[i][6])
                    if((int(data[i][6])) == 0):
                        total_entries = total_entries - 1
                #del data[0]
                print("The final addition is:", add)
                print("The total number of entries are :", total_entries)
                if(total_entries):
                    day_average = add/total_entries
                else:
                    day_average = 0
                day_average = int(hour_average) 
            csv_write('day_logs.csv', 'a',hour_counter, {'YEAR': str(YEAR), 'MONTH': str(MONTH), 'DATE': str(DATE), 'HOUR': str(HOUR), 'MINUTES': str(MINUTES), 'SECONDS':str(SECONDS), 'NUMBER':str(day_average)})
        


        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        
        

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        verbose_classes = [category_index.get(i) for i in classes[0]]

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)
        

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()

