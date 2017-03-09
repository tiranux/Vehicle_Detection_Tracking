##Writeup Vehicle Detection and Tracking

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_nocar.png
[image2]: ./images/hog_features.png
[spatial_bin]: ./images/spatial_bin.png
[color_hist]: ./images/color_hist.png
[mesh]: ./images/mesh.png
[hog_sub]: ./images/hog-sub.jpg
[scales_detection]: ./images/scales_detection.png
[heat_maps]: ./images/heat_maps.png
[labels_boxes]: ./images/labels_boxes.png
[video1]: ./video/project_video_lanes_vehicles.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is random an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  Even though visualizations are good to choose what combination of parameter to use, I have realized that when working with image preprocessing for machine learning algorithms, we cannot always trust our eyes, since the computer does not see the same way we do. So, insted of looking at the result of hog, I really took the decision by simply testing the performance of the SVM with a small set (as int the quiz). Here are a random image with the parameter that I choose from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like, but just for reference. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`which are the parameters that I choose after several small testing:


![alt text][image2]

HOG Features are not enouth to get a good detections, it is also necessary to combine with the other techniques; such as color histogram and spatian bin. 

#### Color Histogram
Color is also an importante feature when detecting cars, given that colors of cars are different from other color that we see in the road; like asphalt, trees, sky, etc.
For color histogram, the features were extracted using YCrCb color space and its parameter is `spatial_size = (32, 32)` . This is how the histogram looks on this color_space for a car image:

![alt text][color_hist]

#### Spatial bins
And finally the features for spatial bins proved to improve the mode, so I extracted features using 32 bins `hist_bins = 32`  and the histogram looks like this:
![alt text][spatial_bin]

You can check these steps in cell 3 of jupyter notebook, where the variables for extractions are initiated and the calls to `extract_features()` are executed. This function return the array of features according to parameter submitted, in our case the three techniques describes before.

####2. Explain how you settled on your final choice of HOG parameters.

I first tried with various combinations of parameters using the quizes from the course. Once I got a good accuracy level (above 98%) I moved to the code in my local implementation and started fine tuning the parameters. I did not really based on what the visualizations look like, as mentioned before, I have realized that we cannot always trust what our human eyes see but we have no mo choice that test different combinations. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog_features on YCrCb channels, spatial binning and histogram of color features. I started with default parameters for Linear SVM and then started trying with different datasets. These are the datasets are tried:
	-original big dataset
    -original small dataset
    -big dataset balanced (same number of car and non-car)
    -big dataset removing similar images
    
Before the training phase, the featurea are scalen with StandardScaler from sklearn:
```
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
The code for data normalization can be found in cell 3 of the notebook.

All dataset splitted in 95% training and 15% testing. I did not see any really improvement on using a balanced dataset nor removing similar pictures, so I went back to the original big dataset. At this point I got around 99.3 % accuracy.

The next thing I tried was to tune the parameters for the model, I tried with various values of C variable. I realized that with a lower C value the accuracy decreases but the false positves are reduced. This evidences that the model tend to overfitting with the default C value of 1.0.  After trial and error on difficult frames I finally chose a C value of 1e-5. I noticed that the randome state variable in the model helps to get a better score. I endend up using a value of 40 for the random seed.

The final accuracy of the model was around 98% which is lower that the original 99.3% but more effective in the final implementation.

I also made some attempts with decision trees but the detection rate is really poor, and with non linear SVM (rbf kernel) the accuracy is better than a linear SVM but the performance is extremely bad. So, I discarded these options and kept the linear one.

The code for the training can be found in cell 4 of the notebook:
```
# Use a linear SVC 
svc = LinearSVC(C= 1e-5,  random_state=40,  verbose=1)
...
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Instead of a sliding window sear in which the features are extracted one by one, I implemented more efficient method for doing the sliding window approach, one that allows us to only have to extract the Hog features once. The function `find_cars()` is able to both extract features and make predictions.

The find_cars only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Its possible to run this same function multiple times for different scale values to generate multiple-scaled search windows.

![alt text][hog_sub]

**Window Scales**

I have chosen these 3 scales based on reasoning and empirical practice: 

**Window Scale 1.2**
	This was chosen to detec the car features when it is smaller (it' farther). Then I realized this was also the most 		time consuming search since it creates more windows. So I realized that is was not neccesary to use this small window 	  in the whole lower half. So I decided to search with this scale on from Y=400 to Y=500. This not only improved speed, 	it also helped to remove a few false positives.
 
**Window Scale 1.4**
  This second scale is chosen to detect average objects. This is a good size for a good   number of detections and reduce false positives. The search is performed from Y=400 to Y=620.

**Window Scale 1.8**
	This scale is the one that detects less object but also does not detect false positives, it helps to increase the whole area of detection once combined with previous scales. The search is performed from Y=400 to Y=656.

![alt text][mesh]

The different window scale cales are executed within the pipeline in cell 10 of the notebook:

```
	ystart = 400
    ystop = 500
    scale = 1.2
    bboxes1 = find_cars(img, ystart, ystop, scale,......)
    
    scale = 1.4
    ystart = 400
    ystop = 620
    bboxes2 = find_cars(img, ystart, ystop, scale,......)
    
    scale = 1.8
    ystart = 400
    ystop = 656
    bboxes3 =  find_cars(img, ystart, ystop, scale,......)
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As described in the previous point, I ended up using three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result in compariston to the use of only one kind of featured. The decicion on searching in the lower half and specifically, only in the regions that make sense acording to the window scale also drastically reduce the processing time. Here are some example images:


![alt text][scales_detection]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a ["link to my video result"](https://youtu.be/BQ682I1ptFk)

Or you can also check it in  [video folder](./video/project_video_lanes_vehicles.mp4)

Here is also one ["video of my own"](https://youtu.be/QuqjMvQ5Nsk). Also in [local video folder](./video/vehicle_detection_own_video.mp4)



I combined this with Advance Lane Detection project by converting it into a class and importing the .py for that project:
`import CarND_Advanced_Lane_Line`

Instantiating the class (cell 11 of notebook):
```
# Instance Advanced Lane Detection Project
laneLinesDetection =  CarND_Advanced_Lane_Lines.AdvancedLaneFinding()
laneLinesDetection.prepareLaneDetection()
```

And simply calling the pipeline within the pipeline of this project  (cell 10 of notebook):
```
# Process frame for lane detection
img =  laneLinesDetection.processLanePipeline(np.copy(image))
```

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have implemented three approaches to remove false positives from the detections. These are described next:

**Removing isolated boxes**

I realized that when a false positive appears in a frame it is usually isolatef from all the true positive detections. So I have created a function called `deleteIsolatedSquares(bbox_intersect)`, which receives as a parameter the list of boxes found by `find_cars()` function. It iterates in the list and check is every box intersects with at least one of the other boxes in the frame. If the box does not intersect with any other box, it is considered a false positive and is removed. The formula employed to detect instersections is as follows:

```
    	 Ax1 <= Bx2  &  Ax2 >= Bx1  & Ay1 <= By2  & Ay2 >= By1
         (x1, y1) = (top_left_corner)
         (x2, y2) = (bottom_right_corner)
```
Where A and B are the boxes being compared.  

You can check the code for this  function in cell 9 in the notebook.

**Hard negative**

I also used hard negative mining, which consists on getting the patches of false positive detections and then add them to the training set. I ran the pipeline on specific frames in where I detected false positives, saved those false positives with a 64x64 resolution, and then trained the model again including those false positives. It helped to remove some false positives like the ones detected on the asphalt.


**Heatmaps**

I recorded the positions of positive detections for the last 3 frames of the video.  From the positive detections I created a heatmap and then thresholded that map with heat value of 6, to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I draw the final bounding boxes to cover the area of each blob detected.  

The threshold value is high because we are combining previous frame detections, wich increase the heat area and will not remove the real detections but the false positives.

Heatmaps are called within pipeline function `pipelineVideo(image)` in cell 10, and the spefic funtions:
```
def add_heat(heatmap, bbox_list):
    ...
    
def apply_threshold(heatmap, threshold):
    ..

def draw_labeled_bboxes(img, labels):
	...
```
Are declared in cell 9 of jupyter notebook.   



   
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are four frames and their corresponding heatmaps:

![alt text][heat_maps]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all four frames and the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][labels_boxes]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem that I issued, was the false positives detection, despicte of the fact that creating the classifier is quite simple, it became a bit difficult to work on tunning. Most of the time that I spent was on mitigating false positives in each frame. I was able to mitigate all false positives related to non cars, but I had a few false positives with real cars coming in opposite direction in adjacent track (which are supposed not to be detected).  

Even though SVM have being used in this area of AI for a while, it is clear that now we have more effective tools, as CNN. In a future implementation I will try with my own CNN or also with existing tools like YOLO. A CNN will is also expected to give a better performance. 



