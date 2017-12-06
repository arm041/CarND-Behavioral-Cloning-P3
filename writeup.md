# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidiaArch.png "Model Visualization"
[image2]: ./examples/normal.jpg "Normal Image"
[image3]: ./examples/rec_left.jpg "Recovery Image"
[image4]: ./examples/rec_right.jpg "Recovery Image"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 the video output of the simulated model model.h5 I used on the simulator

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### General explanation about Model Architecture and Training Strategy

The model I use in this project is the model introduced by NVIDIA for self driving cars. This model consist of first a normalization layer that normalizes the inputed data. After this layer I implemented 5 convolution layers, from which the first three have a filter size of 5x5 and the last two of the filter size of 3x3. After that the CNN is flattened and consists of 4 fully connected layers. This model is implemented in model.py in lines 61-75. 
The model includes RELU layers to introduce nonlinearity (code line 64-69), and the data is normalized in the model using a Keras lambda layer (code line 62). 
The model contains dropout layers in order to reduce over fitting (model.py lines 71 and 74). 
The model was trained and validated on different data sets to ensure that the model was not over fitting (code line 78). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, also to generalize the model I drove counter clock-wise in the simulator. Also I drove 1 track through the challenging map of the simulator to gather more data.   

### Model Architecture and Training Strategy

The overall strategy for deriving a model architecture was to first start with a well known model and check how it performs on this examples here. The most easy model I could think of was just a convolution layer which work very poorly but help me to setup the environment and learn how to use the simulator and gather and use the data to train a model with Keras. 
My next step was to use the LeNet architecture in this problem to understand how this layer would perform here. I thought his model might be appropriate because LeNet model works generally fine for image processing and also I used it before for traffic sign classification with success.
In order to understand how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Also for the LeNet architecture this phenomena was observed. 
Before combating this problem I changed my model to a model by NVIDIA for self driving cars, and added drop out layers to this model to fight with the overfitting problem in the training set.  
After getting satisfying results on the validation set the final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve this behavior I started to collect related data for this situations like driving .. to improve the driving behavior in these cases, I intentionally drove to the end of lines and went back to the center such that the model can learn these situations and steer back to the center of the image.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
The final model architecture (model.py lines 61-75) consisted of a convolution neural network with the following layers and layer sizes:

* Normalization layer which is a keras lambda layer
* A Cropping layer to cut some pixels from the image
* Convolution layer with filter size 5x5 and output width of 24
* Convolution layer with filter size 5x5 and output width of 36
* Convolution layer with filter size 5x5 and output width of 48
* Convolution layer with filter size 3x3 and output width of 64
* Convolution layer with filter size 3x3 and output width of 64
* Fully connected flatten layer with 100 outputs
* Drop-out layer with keep probability of 70 percent
* Fully connected flatten layer with 50 outputs
* Fully connected flatten layer with 10 outputs
* Drop-out layer with keep probability of 70 percent
* Fully connected flatten layer with 1 outputs

Here is a visualization of the architecture:

![alt text][image1]

The only difference is that the image above is taken from the NVIDIA website and here I used an input with a different size. My input is from size (160,230,3) which is then cut to (65,230,3). 
To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it gott close to the lane lines and also how to navigate through curves. These images show what a recovery looks like starting from recovery from left side lane and right side lane:

![alt text][image3]
![alt text][image4]


Then I drove once center lane through the second track to get more data points.

After the collection process, I had 16264 number of data points. I then preprocessed this data by cutting of the pixels that are consisting more of trees and the sky and also the down pixels that show the car, so 70 pixels from the top and 25 pixels from the bottom. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph of mean squared errors also produced in the model.py file in lines 84-90. I used an adam optimizer so that manually training the learning rate wasn't necessary.
