# Plant-Wheat-Detection-Using-Ai
This project analyzes the detection and the growth stages of wheat crop by capturing a digital image of the crop from time to time by a Raspberry pi Camera Module V2. This image is then transferred to a computer for analysis. The analysis is done by the deep learning algorithm Convolutional neural network CNNs 

2 Main Architectures for CNNs used Fast RCNN and Faster RCNN:-

Fast RCNN 

we feed the input image to the CNN to generate a convolutional feature map. From the convolutional feature map, we identify the region of proposals and warp them into squares and by using a RoI pooling layer we reshape them into a fixed size so that it can be fed into a fully connected layer. From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the offset values for the bounding box.

![1](https://user-images.githubusercontent.com/64171873/174430999-fcd30ae7-290b-4da7-9c9c-6587649788ba.jpg)

Faster RCNN

using selective search to find out the region proposals. Selective search is a slow and time-consuming process affecting the performance of the network.
So Faster RCNN eliminates the selective search algorithm and lets the network learn the region proposals.Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI pooling layer which is then used to classify the image within the proposed region and predict the offset values for the bounding boxes.

![1](https://user-images.githubusercontent.com/64171873/174431126-21112094-c23c-4e0c-9f76-80dd3f9ddd5c.png)


Here are some of projects results :- 

![Result4](https://user-images.githubusercontent.com/64171873/174431195-4b0d0c6c-78b3-44f7-9670-7092f2864270.jpg)

![Result5](https://user-images.githubusercontent.com/64171873/174431199-34532c4e-430d-4775-b63b-eb7379616cd1.jpg)

![Result1](https://user-images.githubusercontent.com/64171873/174431200-907e755d-0e38-4f71-bb81-2ebcc1bea384.jpg)

![Result2](https://user-images.githubusercontent.com/64171873/174431202-112dd8a7-b62b-4dfa-acf3-256a77ec335d.jpg)

![Result3](https://user-images.githubusercontent.com/64171873/174431203-5fd16e28-6a03-4dd9-90ed-556b72f84623.jpg)

