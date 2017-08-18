## Project Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_resources/before_calibration.jpg "before calibration"
[image2]: ./writeup_resources/after_calibration.jpg "after calibration"
[image3]: ./test_images/straight_lines1.jpg "Distorted image"
[image4]: ./writeup_resources/undistorted.jpg "Undistorted image"
[image5]: ./writeup_resources/edge/test1.jpg "Edge detection example"
[image6]: ./writeup_resources/warped.jpg "Warped"
[image7]: ./writeup_resources/edges_and_warped.jpg "Edges & Warped"
[image8]: ./writeup_resources/curvature_formula.jpg "Curvature Formula"
[image9]: ./output_images/straight_lines2.jpg "Final output"
[video1]: ./output_images/project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Camera class under the
`calibrate_camera` method.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `img_obj_points` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
![alt text][image2]

The distortion correction method is using all the images in the camera calibration folder in order to minimize the camera distortion
as much as possible.

### Pipeline (single images)
Before The image processing begins, an Camera instance is created and calibrated, and the lane lines class, which stores infp about the lines, is also initialized for both lines.
#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

In the pipeline, the image camera distortion is fixed as part of the `camera.get_lane_view(image)` method. This method is using the camera matrix and distortion coefficients that were calculated when the pipeline started.

The end resulted image after camera distortion is fixed is
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
This is the second step that is done in the `campera.get_lane_view(image)` method.

The code that uses the color thersholds to create a binary image is found at the `camera._get_edges()` method.
This method expects to get in addition to the image, three tuples that contains low and high thresholds:
* `s_thresh` - Those thresholds are used on the S-channel of the image after HLS conversion. This threshold purpose is to help detect the white (and some of the yellow) lane lines in the pictures.
* `sobel_thresh` - Those thresholds are used on the magnitude of the of the L-channel x-y normalized derivative. This threshold purpose is to help detect the lines under shadows.
* `red_thresh` - Those thresholds are used on the R-channel of the image (RGB image). This threshold purpose is to help detect the yellow lane lines in the pictures.  

I used a combination of color and gradient thresholds to generate a binary image (thresholding is done by the `camera._create_bit_mask()` method).

After creating all three bit masks the `camera._get_edges()` method is returning a bit mask that is composed from all three bit masks by applying bitwise or on all three masks.

Here's an example of my output for this step.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
This is the third step that is done in the `camera.get_lane_view(image)` method.

The code of my prespective transform starts at the Camera class `__init()__` method, where the transformation source & destination points are hardcoded like so:

```python
# Bird-eye wrap preparation
# Source points(formatted like the actual rectangle)
src_points = np.float32([[594, 450], [688, 450],
                         [200, 720], [1100, 720]])
# Create the destination points array
dst_points = np.float32([[300, 0], [1000, 0],
                         [300, 720], [1000, 720]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 594, 450      | 300, 0        |
| 200, 720      | 300, 720      |
| 1100, 720     | 1000, 720      |
| 688, 450      | 1000, 0        |

After that the warped matrix and the inverted wrap matrix are being calculated and set to `camera._wrap_mat` and `camera._wrap_mat_inv` respectively.

The transform is done in the `camera.get_lane_view(image)` method by using  `cv2.warpPerspective()` and the pre-calculated wrap matrix.

Here an example for an image after the wrap transform:
![alt text][image6]

In the pipeline the wrap is done after the edge detection step so the image will actually will look like so:
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane line detection and fitting can be divided into two main phases: lane line detection and line smoothing.
In still images only the line detection phase is active.

##### a. lane line detection
The wrapped image is passed to the `detect_lane()` method [line 282] which works differently for still images and videos (by setting the flag `movie_mode`).
On still images this method uses an internal method called `find_lane()` which searches for the lane line by using the sliding windows method that was presented on the project section in Udacity:
* Calculate binary image lower half histogram in order to get the initial searching point for the lines.
* Divide the image into windows when `window_height=image_height/num_of_windows`.
* For each window in the left and right lines count the number of none zero pixels and if they suppress the `recenter_thresh`, re-center the window by averaging the detected none-zero pixels.
* After going over all the windows, use the collected none-zero pixels and calculated the line 2nd rate polynomial coefficients.
* validate the fitted line via the `validate_lane_lines()` helper method (will be explained afterwards).
* If validation passed, update the `current_fit`, `all_x`, `all_y` and `diffs` properties of the lines.

The validation method that was used is validating the lines by checking the variance of the distance between them and by checking that do not cross by checking that the distance between the right line to the left line is always positive.

The behavior of the lane line detection phase and the smoothing phase in `movie_mode=True` will be explained at the video pipeline section.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code who calculates the lines curvature and position with respect to center is found at the `update_curvature_and_pos()` method [line 516].

In order to calculate the above values the detected pixels need to be converted to meters (in order to represent real world information), this was done by using some knowledge on lane lines in America. We know what should by the distance between two lane lines and also we can calculate the distance of the camera line of sight and use those in order to convert pixels into meters.

Then. by using the detected pixels and multiplying them by the conversion factor, a polynomial that represent the "real" lane lines can be calculated.

After the polynomial was calculated curvature can be calculated by using the formula: ![alt text][image8]

And the relative position is calculated by evaluating the polynomial at the bottom of the image and measuring the distance of that point from the center of the image (multiplied by the conversion factor).


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `camera.draw_on_image()`
Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Changes in lane line detection for video pipeline

As I mentioned above, The lane line detection and fitting can be divided into two main phases: lane line detection and line smoothing.

##### a. lane line detection
In addition to the lane line finding method `find_lane()`, the video pipeline uses anther method called `track_lane()`.

After the lane lines were detected, in most of the cases it sufficient to use the previous fit in order to find them in the next frame. That works because the lanes are continuous and the overall position of the lane lines in the video should not change much.
The tracking algorithm is also the same as the one presented on the project section in Udacity:
* Calculate the x values of the previously fitted lines.
* Look for any none-zero pixels at a pre-configured margin from the lines locations.
* Use the pixels that were found to calculate the new polynomial coefficients and validate it using the validation method.
* If validation passed, update the `current_fit`, `all_x`, `all_y` and `diffs` properties of the lines.

On each frame, the detection method tries to track the lines first (The only exception is the first frame of the video which gets the same treatment as a still image).
If tracking fails the detection algorithm decides if to skip the detection process and use the previous fit or to use the finding algorithm, this decision is made based on the value `max_skip`.

##### b. lane line smoothing
This phase is implemented on the `average_lines()` method.
This phase purpose is to create smoother transitions between frames.
The method to do so is by using weighted average on the x points of the fitted line: `best_x=(1-alpha)*best_x + alpha*current_x`, when alpha is between zero to one.
When using high alpha values, the current fit affect the result more.
After the re-calculation of `best_x`, we calculate the best fit polynomial coefficients.

#### 2. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the problems I encountered were in the edge detection part of the project. Different environments and road conditions make the problem of finding the optimal thresholds harder. At some images especially in the challenge videos, the image became almost undetectable while in the project video the edges were clearly visible.

My pipeline will likely fail at images that are not well lit, images in bad weather (rain, fog) and images with faded lane lines. Anther point of failure can be change in the visibility range of the road, when the horizon line is lower than the maximum point of the region of interest the pipeline can detect "phantom lane lines". That can happen when the vehicle is ascending.

In order to make the pipeline more robust, we can adjust edge detection parameters based on the environment - create edge detection presets and choose the optimal preset.
Use the detected lane lines as feedback for changing region of interest and to adjust other detection related parameters.
