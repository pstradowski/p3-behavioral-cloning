
# **Project Report: Behavioral Cloning** 

## Udacity Self Driving Car Nanodegree 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/histogram.png "Histogram of Udacity dara"
[image3]: ./images/histogram_nonzero.png "Histogram with zero values removed"


---
### Files in the repository
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

### Model Architecture and Training Strategy

#### 1. Model architecture

My model uses network developed by Vivek Yadaw, it uses filter size of 3, ELU activations and dropout layers to reduce overfitting. More detailed description and reason for choosing this architecture is explained in next chapters.

#### 2. Attempts to reduce overfitting in the model

My approach to fight overfitting was based on 2 methods:
* Use dropout in the model with default weight 0.5
* split dataset int train and validation and check loss on the validation datasef

While dropout works as usual, the validation dataset did not yield expected results. Usually the model, which drove best in simulator, had not the best mean square error. This could be caused by nature of the steering signal.



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The dataset was augmented in amy ways so it had more impact on the end result than model parameters, see

#### 4. Appropriate training data

I used 2 data sets: the official one released by Udacity and one recorded by me. Both sets have similar length - ~8000 pictures and my challenge was to generalize model using only this data. In final version, I used only Udacity data with augmentation and steering distribution shaper.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My initial goal was an architecture which should be as simple as possible, therefeore I started with comma.ai model, then switched to Nvidia architecture. I tuned both architectures heavily and found Nvidia base better than comma.ai, however this should be double checked.
I took into account transfer learning on VGG16 from Image Net, however the first good results turned me to tune another model. 

One of my assumption was to allow for quick trial and error, therefore I needed something which I can train fast using either my laptop or AWS GPU instance.
In the end, I found Vivek Yadaw network, which contains an additional layer to deal with colorspaces and it performed better than all my experiments.

Since beginning, I worked with train and validation set, however did not notice an overfitting - mainly due to heavy usage of dropout layers in my networks.
One of most important outcomes for me is that loss function does not give us trusted results. It could be caused by many factors but the only remedy I found was to test models with simulator and observe its behavior.
Most difficult part for me was the choice of augmentation and shaping the steering distribution - I developed Keras generator with pluggable feeders - this allowed me to change distribution shape often and observe results. Feeders shaped the steering distribution using different approaches, however the most effective one was a simple binning algorithm called randliner.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I did some speed tuning and 20 seems to be max speed the model can drive on Track 1. See the youtube video below for a demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=YBAEEdjm_V0
" target="_blank"><img src="http://img.youtube.com/vi/YBAEEdjm_V0/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>
#### 2. Final Model Architecture

After addiotinal augmentation with shadow data, the model is able to complete Track 2
The final model architecture is included in function vivek_model (line 208)

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Due to my personal challenge, I decided to not use an additional dataset, the whole training was made on the Udacity one. One of my rationales was my lack of gaming skills to play with simulator and together with opinions form our Slack channel this caused me to use limited dataset.
Input data is unbalanced and skewed - there is a lot of zero values, this balance of data makes learning difficult due to high bias towards not taking turns:

![alt text][image2]

After removing zero values, we can observe the distribution shape of the remaining data:

![alt text][image3]

The model is trained using Keras generator. The generator uses an object called feeder, which gives the generator the file name and steering for further processing. This gave me possibility to experiment with different shapes of steering angle distribution and tune the learning process.

I experimented with multiple augmentation techniques:
* Left, right and center camera were choosen randomly
* Balance the left and right tuns, I used random filipping

My main image processing pielinepipeline includes following elements:

1. Random selection of left/right/center camera and appropriate steering correction +/- 0.25 
2. Random brightness correctiom
3. Random flipping and steering reversing - this part removes skeweness from the data.
4. Adding shadows to the image
5. Crop 60 pixels from the top and 20 from the bottom to remove unecessary information and give the network only these part of picture, which contain inforation used to take decision about steering. It was a suprise for me that cropping yields so much improvement for the model.
6. Resize to 64x64 pixels


#### 4. Summary
This was most challenging project in this Nanodegree, mainly to model sensivity to steering samples distribution. Finding the right distribution took me lot of time and combinantion with other model parameters like augmentation made this process more complex. One of mistakes, I made was complex solution at the beginning. Paul Heraty suggested very simple approach, and if I started this way I would find a solution in shorter time. This project showed, that Deep Learning is very empirical - thre is no one receipe to deliver good model and lot of knowledge came from trial and error in simulator.

