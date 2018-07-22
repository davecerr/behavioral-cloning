# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives once complete lap around the track without leaving the road



[//]: # (Image References)

[image1]: ./training.png "Training and Validation Loss"
[image2]: ./cropping.png "Normalisation, Mean-Centering and Cropping"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* completed_run.mp4 video of central dashboard camera during the completed lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture is as follows:

| Layer Description                       |Input    |Output   |# Trainable Parameters|
| --------------------------------------- |---------|---------|----------------------|
| Input                                   |160x320x3|160x320x3|0                     |
| Normalise & Mean-Centre                 |160x320x3|160x320x3|0                     |
| Cropping                                |160x320x3|70x240x3 |0                     |
| 5x5 Convolution with ReLU (64 filters)  |70x240x3 |66x236x64|4864                  |
| 2x2 MaxPooling                          |66x236x64|33x118x64|0                     |
| 5x5 Convolution with ReLU (16 filters)  |33x118x64|29x114x16|25616                 |
| 2x2 MaxPooling                          |29x114x16|14x57x16 |0                     |
| 5x5 Convolution with ReLU (4 filters)   |14x57x16 |10x53x4  |1604                  |
| 2x2 MaxPooling                          |10x53x4  |5x26x4   |0                     |
| Dropout (keep_prob=0.1)                 |5x26x4   |5x26x4   |0                     |
| Flatten                                 |5x26x4   |520      |0                     |
| Dense 1                                 |520      |100      |0                     |
| Dense 2                                 |100      |16       |0                     |
| Dense 3                                 |16       |1        |0                     |


This has a total of 85,917 trainable parameters.


#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. This is demonstrated in the accompanying video file, "completed_run.mp4".

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I thank Eric Lavigne for helping me obtain suitable training data. The important step here was to realise that large amounts of training data is not required. Instead, using just 107 carefully chosen track positions and accompanying steering angles, we can get 321 images (one for left, right and centre) and train the model on that. This has the benefit of training much faster and not requiring a GPU, as well as being sufficiently small that once the car is identified as failing at a specific location in the simulator, we can simply go to that location and record a new training image to recitfy the problem. This approach would not be possible with a very large dataset since adding a single image to fix the problem would be insignificant next to the huge amount of pre-existing data.

As can be seen in model.py, I make use of images from all three cameras (left, right and centre). For the left and right camera images, a correction factor of +/- 0.1 was applied to the steering angles to account for these cameras not occupying a central position on the dashcam. This will help the vehicle maintain a central road position. I also flip the image from the central camera and reverse its steering angle. This makes the data less biased to turning left (bear in mind the car has been trained driving anticlockwise around a circular track so there is an inherent bias).

To preprocess the images, they were first normalised and mean centred. I then cropped by removing the top 70 pixels (skyline not relevant), bottom 20 pixels (car bonnet not relevant), as well as the leftmost and rightmost 40 pixels (with the idea being this would make the model more focussed on the road ahead):

![alt text][image2]

After the collection process, I had 428 data points. I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 25 epochs and used callbacks to keep the best model i.e. the model from the epoch with the lowest validation accuracy.

#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to apply a single convolution layer and then build a typical CNN pyramid structure (convolution layer followed by pooling layer) of increasing depth until performance reached a suitable level that I could turn my attention to tuning hyperparameters. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I decided to use mean absolute error rather than mean squared error. Comparing the two loss functions, mean squared error will penalise more heavily when there is a large difference between the predicted steering angle and the actual steering angle. Thus, it is safest for the model to just learn to predict 0 all the time and indeed, I found my model to be very enthusiastic about driving in a straight line when using mean squared error. With mean absolute error, this is not the case and the model is more willing to make predictions about large steering angles that deviate from zero (straight ahead). I found the model quickly improved its ability to handle cornering.

My model initially had a high validation loss indicating it was overfitting the training set and so I added a dropout layer to prevent this. I now have a model with a very low validation loss compared to the training loss. In fact, the training loss remains quite high and this possibly indicates there is some underfitting going on. If I had more time, I would try to make a more complicated model that reduces this training loss. However, I am pleased to have built a model that can drive a lap of the track and so am happy to stop here.

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is now able to drive autonomously around the track without leaving the road. It is interesting to watch it as it crosses the cobbled road surface on the bridge - it starts to weave around a lot more. This is because the rest of the track is tarmac and so this unusual surface causes some confusion. Whilst the model can handle it, and does successfully cross the bridge, I feel it is worth commenting on.



