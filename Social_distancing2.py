from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy
from scipy.spatial import distance as dist
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def midpoint(coordinates):
    y = int((int(coordinates[2]*imH) - int(coordinates[0]*imH))/2)
    x = int((int(coordinates[3]*imW) - int(coordinates[1]*imW))/2)
    return tuple((x,y))

def width_calc(coordinates):
    w = int(coordinates[3]*imW) - int(coordinates[1]*imW)
    return w

def distance_calculation(boxes_primary,boxes_secondary):
    for primary in boxes_primary:
        w_primary = width_calc(primary)
        midpoint_primary = midpoint(primary)
        for secondary in boxes_secondary:
            midpoint_secondary = midpoint(secondary)
            w_secondary = width_calc(secondary)
            if (w_primary - w_secondary) >50:
                pass
            else:
                dist_px = dist.euclidean(midpoint_primary,midpoint_secondary)
                dist_ft = (dist_px/w_primary)*1.167
                print(dist_ft)
                if 0 < dist_ft < 0.2 :
                    #coordinates of the primary box
                    ymin_primary = int(max(1,(primary[0]*imH)))
                    xmin_primary = int(max(1,(primary[1]*imW)))
                    ymax_primary = int(max(1,(primary[2]*imH)))
                    xmax_primary = int(max(1,(primary[3]*imW)))
                    #coordinates of the secondary box
                    ymin_secondary = int(max(1,(secondary[0]*imH)))
                    xmin_secondary = int(max(1,(secondary[1]*imW)))
                    ymax_secondary = int(max(1,(secondary[2]*imH)))
                    xmax_secondary = int(max(1,(secondary[3]*imW)))
                    #plotting rectangles in red color
                    cv2.rectangle(frame, (xmin_primary,ymin_primary), (xmax_primary,ymax_primary), (0, 0, 255), 2)
                    cv2.rectangle(frame, (xmin_secondary,ymin_secondary), (xmax_secondary,ymax_secondary), (0, 0, 255), 2)


resW = 500
resH = 268
imW,imH = resW,resH


interpreter = tflite.Interpreter(model_path = 'detect.tflite')
interpreter.allocate_tensors()

#getting model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#check the type of input tensor
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
boxes_primary = []
boxes_secondary = []
    # ret,frame = cap.read()
frame = cv2.imread('2.jpg')
    #loading image in 'frame' variable
frame = cv2.resize(frame,(imW,imH))
frame_resized = cv2.resize(frame,(width,height))
input_data = np.expand_dims(frame_resized,axis = 0)

if floating_model:
    input_data = (np.float32(input_data)-input_mean)/input_std

    #performing detection on an image
interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()

    #retriveing detection results
boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
for i in range(len(scores)):
    if classes[i] == 0:
        if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            boxes_primary.append(boxes[i])
            boxes_secondary.append(boxes[i])

distance_calculation(boxes_secondary,boxes_secondary)
cv2.imshow('frame',frame)
cv2.waitKey(0)
