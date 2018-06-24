# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[right]: ./img/rightCamera.jpg "Right Camera Image"
[center]: ./img/centerCamera.jpg "Center Camera Image"
[left]: ./img/leftCamera.jpg "Left Camera Image"
[centerDriving]: ./img/centerDrive.jpg "Recover From Left"
[centerDrivingFlipped]: ./img/centerDriveFlipped.jpg "Flipped Image"
[rightSteered]: ./img/recoveringFromRightCorner.jpg "Steered Right"
[leftSteered]: ./img/recoveringFromLeftCorner.jpg "Steered Left"
[recoveredRight]: ./img/recoveredFromRight.jpg "Recoverd From Right"
[recoveredLeft]: ./img/recoveredFromLeft.jpg "Recoverd From Left"
[cropped]: ./img/cropped.jpg "Cropped Image"
[meanError]: ./img/MeanSquareError.png "Mean Square Error"
[cnn]: ./img/cnn_nvidia.png "Mean Square Error"

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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a high mean squared error on the training set but a low mean squared error on the validation set. This implied that the model was not overfitting. 

![alt text][meanError]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model architecture is borrowed from Nvidia's CNN for SDC ([model.py lines 99-112](model.py#L99-L112)). 
The original architecture ([Click for details.](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "Click for details.")) is shown in the following figure. 

![alt text][cnn]

I've done following modifications to adjust to my requirements:
* A cropping layer was added right after Normalization to crop 50 pixel from top and 20 pixel from bottom. This will help remove the unwanted area from the image and learning is more relevent.
* Additional layer towards the end to have single output

#### 3. Attempts to reduce overfitting in the model

* I split my  data into 80% training and 20% validation data. Spliting 80:20 is consider a good thumb of rule for training purpose
* Kept the epoches count low to **3** and batch size to **32**

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py line 25](model.py#L115)).

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

#### 6. Creation of the Training Set & Training Process

The simulator captures the data using 3 cameras mounted on left, center and right side of the car. This helps the capture data in different angles covering all necessary paths across the drive lane

Output from left, center and right camera:

![alt text][left]
![alt text][center]
![alt text][right]

To capture good driving behavior, I recorded a laps on track one using a mix of center lane driving and recovering from corner. Here is an example image of center lane driving:

![alt text][centerDriving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover for corner driving. 

These images show what a recovery looks like starting from

<pre>          Steered to Left                         Steered to Right </pre>

![alt text][leftSteered]
![alt text][rightSteered]

<pre>         Recoverd from Left                      Recoverd from Right </pre>
![alt text][recoveredLeft]
![alt text][recoveredRight]

To augment the data sat, I also flipped images and angles thinking that this would help avoid  For example, here is an image that has then been flipped:

![alt text][centerDriving]
![alt text][centerDrivingFlipped]

A cropped Image after Cropping2D layer

![alt text][cropped]

After the collection process, I had 3891 number of data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation error doesn't improve. I kept the batch size comparetively small to 32. I used an adam optimizer so that manually training the learning rate wasn't necessary.


A link to my youtube video for [Track 1](https://youtu.be/k8vlqTaB4sc "Track 1")

### Future Enhancements

* Enhance model.py to add capability to iterate over different data set in one go. Currently it iterates for one folder only
* It doesn't work very well with Track2. Train model using data from track2   
* Augment images to add some shadow efffect this will enhance roboustness for track2.
