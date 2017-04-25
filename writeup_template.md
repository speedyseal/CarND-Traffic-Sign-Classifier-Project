#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[traininghist]: ./p2img/traininghist.png "Training set histogram"
[validhist]: ./p2img/validhist.png "Validation set histogram"
[testhist]: ./p2img/testhist.png "Test set histogram"
[postbalancehist]: ./p2img/postbalancehist.png "Training set histogram after balancing"
[stoppreprocessed]: ./p2img/stoppreprocessed.png "Preprocessed stop sign from original set"
[stopaugmented]: ./p2img/stopaugmented.png "Stop sign from augmented set showing zoom and rotation"
[image4]: ./webimages/c22.jpg "Bumpy road"
[image5]: ./webimages/c28.jpg "Children crossing"
[image6]: ./webimages/c35.jpg "Ahead only"
[image7]: ./webimages/c38.jpg "Keep right"
[image8]: ./webimages/c4.jpg "Speed limit 70km/h"
[confusionmatrix]: ./p2img/confusionmatrix.png "Confusion matrix"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

My [project code](https://github.com/speedyseal/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-HSV-balance-tuneL2.ipynb) is available on github at the link.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Plotted here are the class histograms for the training, validation, and test sets showing the disparity between class frequencies. Classes with greater representation will have a greater weight in training unless the frequency is balanced across classes. There are various techniques to deal with this. [http://www.ele.uri.edu/faculty/he/PDFfiles/ImbalancedLearning.pdf](http://www.ele.uri.edu/faculty/he/PDFfiles/ImbalancedLearning.pdf)

![alt text][traininghist]
![alt text][validhist]
![alt text][testhist]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted the images to HSV in a separate preprocessing step because doing it inline in a pipeline is too slow on my machine. This is loaded as new pickled data files.

The preprocessing pipeline comprises the following steps:
 - convert to HSV
 - Equalize the histogram on the luminosity channel using cv2.equalizeHist(img[:,:,2])
 - Normalize by subtracting 128 from each channel and dividing by 128 to center the 8 bit uint_8 value around 0 with a range of +/- 1
 
I decided not to convert to grayscale because, unlike in the mnist test set which is color invariant, color is a part of the class feature for street signs. A stop sign that is green is not a valid stop sign on the street. Color helps to distinguish between classes and I think converting to grayscale throws some of these distinguishing features.

As a last step, I normalized the image data because it improves convergence.

I decided to generate additional data because I have poor validation accuracy on the underrepresented classes and the gap between training and validation indicates that the convnet is overfitting to the existing training set.

To add more data to the the data set, I used Keras' image generation function to randomly
 - rotate up to 10 deg
 - skew up to 3 deg
 - zoom up to 5%
 - shift horizontally and vertically up to 10%
 - randomly shift channel data by 2

I use the randomly transformed images to augment each class independently until each of the classes has an equal total number of samples. The number of samples per class in the training set can be specified as a parameter.

This image shows samples from the original set next to samples from the augmented set.
![alt text][stopaugmented]

The difference between the original data set and the augmented data set is illustrated in the following histogram showing that all classes have more or less the same number of samples each.
![alt text][postbalancehist]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  					|
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| dropout				| keep prob=0.5									|
| Fully connected		| input 84, output 42							|
| Softmax				|         										|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer with a rate of 0.001, L2 regularization on the fully connected weights with a strength of 0.001, batch size 128 and 40 epochs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 97%
* validation set accuracy of 97%
* test set accuracy of 93%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I chose the LeNet-5 because I think the data set is too small for a deeper, state of the art net such as VGG or resnet. I am already seeing overfitting on LeNet-5 and I think larger nets will also be quite overfitted. The larger nets tend to be used on large pretrained data sets and then fine tuned on the application set. The advice is "don't be a hero" trying to invent new architectures. I see this as more of an exercise in preparing  data statistics for classification. LeNet works well on images of this size. Some tweaks could be made to account for the increased number of classes vs. MNIST and the usage of 3 color channels. However increasing the number of parameters increases the likelihood of overfitting.

The good match between training and validation sets indicating low overfitting on these two sets, however, the convnet is still overfitted to the combination of these two sets as indicated by the lower accuracy on the test set. More work could be done trying to reduce overfitting.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

These images may be difficult to classify because of parallax errors that aren't present in the training set, or backgrounds or shadowing that are different.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Children crossing     | Children crossing 							|
| Ahead only			| Ahead only									|
| Keep right      		| Keep right					 				|
| Speed limit 70km/h	| General caution      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The quality of speed signs data in the training set may be quite noisy, therefore the classifier may have been unable to find separation between the classes and instead sees the red circle outline, interpreting it as general caution.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children crossing   							| 
| .0     				| Right of way at next intersection 			|
| .0					| Pedestrians									|
| .0	      			| Dangerous curve to right		 				|
| .0				    | Road narrows on right      					|

For the second image, it somehow got very confused. The distorted circle looks more like an oval and the classifier is matching it more towards triangular signs than circular signs.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.51         			| General caution   							| 
| 0.48     				| Traffic signals 							|
| .005					| speed limit 70km/h							|
| .004	      			| No vehicles		 				|
| .002				    | Pedestrians      					|

For the keep right sign, Bumpy road, and ahead only, the classifier is very sure about its predictions. The secondary softmax probabilities are smaller than 1e-5.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep right   							| 



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Bumpy road   							| 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead only   							| 


####4. Confusion matrix

Another standard metric for evaluating a classifier is the confusion matrix. Pandas_ml provides useful utilities to compute and plot the confusion matrix indicating which classes are mistakenly classfied as another.

![alt text][confusionmatrix]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The activations of the first convolutional layer show what features are highlighted. Salient features that cause activations are the outlines of the signs, indicating some kind of edge detection filter learned by the convnet.
It is difficult to interpret the activations of the second convolutional layer since they are essentially transpositions of the first convolutional layer, but there is a nonlinear pool operation stuck in between. To interpret them, one could convolve the filters of the first layer with the activations of the second layer to see what the activations map to in the original image domain.
