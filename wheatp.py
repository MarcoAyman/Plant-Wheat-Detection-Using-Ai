import numpy as np
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
print("Loading Libraries")

model_path = '/home/pi/wheat/inceptinV2_frozen_inference_graph.pb'
test_images_path = '/home/pi/wheat/test/'
sample_csv = '/home/pi/wheat/sample.csv'

print("Loading Paths")

sub = pd.read_csv(sample_csv)
sub.head()
print("Sample CSV File")




with tf.compat.v1.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    
    
    
    
submission = pd.DataFrame(columns=list(sub.columns))
plt.figure(figsize=(20,10))
with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

#         img_name = image_name
        # Read and preprocess an image.
        img = cv2.imread('test1.jpg')
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (640, 480))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                        feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Saving detected bounding boxes.
        num_detections = int(out[0][0])
        pred_str = ''
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                xmax = int(bbox[3] * cols)
                ymax = int(bbox[2] * rows)

                cv2.rectangle(img, (x, y), (xmax, ymax), (0,0,255), 2)
                cv2.imwrite('result.jpg')
                pred = '{} {} {} {} {} '.format(np.round(score,2),x,y,xmax-x,ymax-y)
                pred_str = pred_str + pred
        