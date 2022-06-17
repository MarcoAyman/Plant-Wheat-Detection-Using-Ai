import numpy as np
import tensorflow as tf
import cv2 as cv
from collections import deque
import argparse
import imutils
import urllib
import time

ap = argparse.ArgumentParser()
# ap.add_mutually_exclusive_group("-b", type=int, default=64)
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
 




print("Load Main Libraries")
cap = cv.VideoCapture(0)
# Read the graph.
with tf.compat.v1.gfile.FastGFile('inceptinV2_frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    print("Load Model File")

with tf.compat.v1.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
#     img = cv.imread('test5.jpg')
    while True:
        ret, img = cap.read()
       
       
        frame = imutils.resize(img, width=300)
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        for key, value in upper.items():
            kernel = np.ones((9,9),np.uint8)
            mask = cv.inRange(hsv, lower[key], upper[key])
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            if (key=='yellow' or key == 'green'):
 
 
 
                 
                        rows = img.shape[0]
                        cols = img.shape[1]
                        inp = cv.resize(img, (300, 300))
                        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

                        # Run the model
                        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                        sess.graph.get_tensor_by_name('detection_scores:0'),
                                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                                        sess.graph.get_tensor_by_name('detection_classes:0')],
                                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                        # Visualize detected bounding boxes.
                        num_detections = int(out[0][0])
                        d=0
                        for i in range(num_detections):
                            classId = int(out[3][0][i])
                            score = float(out[1][0][i])
                            bbox = [float(v) for v in out[2][0][i]]
                            
                            if score > 0.3:
                                d=d+1
                                x = bbox[1] * cols
                                y = bbox[0] * rows
                                right = bbox[3] * cols
                                bottom = bbox[2] * rows
                                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                        print("Wheat Detections = ",d)
            
                       
        cv.imshow('Wheat Detection Result', img)
        time.sleep(0.2)
#     cv.waitKey(1)
        if cv.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
img.release()
cv.destroyAllWindows()
