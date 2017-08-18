"""
This script is used for detecting lane line in images
and videos.
"""
import cv2
import os
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# ###### Constants ############
LANE_HEIGHT_METERS = 30
LANE_WIDTH_METERS = 3.7

# The distance in pixels between lanes
LANE_WIDTH_PIXELS = 700


class Line:
    """
    This class contains line info
    """

    __slots__ = ['detected', 'recent_fitted_x', 'best_x', 'best_fit',
                 'current_fit', 'radius_of_curvature', 'line_base_pos',
                 'diffs', 'all_x', 'all_y']

    def __init__(self):
        # Was the line detected in the last iteration?
        self.detected = False
        # Average x values of the fitted line over the last n iterations
        self.best_x = 0
        # Polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # Polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None
        # Difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.all_x = None
        # y values for detected line pixels
        self.all_y = None

    def get_points(self):
        """
        Returns a Tuple with `all_y` and `all_x`
        """
        return self.all_y, self.all_x


class Camera:
    """
    This class handles all camera related functionality.
    """

    __slots__ = ['_camera_mtx', '_camera_dist', '_wrap_mat', '_wrap_mat_inv']

    def __init__(self):
        self._camera_mtx = None
        self._camera_dist = None

        # Bird-eye wrap preparation
        # Source points(formatted like the actual rectangle)
        src_points = np.float32([[594, 450], [688, 450],
                                 [200, 720], [1100, 720]])
        # Create the destination points array
        dst_points = np.float32([[300, 0], [1000, 0],
                                 [300, 720], [1000, 720]])
        # Create wrap matrix
        self._wrap_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        self._wrap_mat_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    def calibrate_camera(self, cal_images_path, chessboard_size):
        """
        Calibrates the camera using chessboard pattern calibration.
        :param cal_images_path: The path for a folder containing
                                camera calibration images.
        :param chessboard_size: The 2D dimensions of the chessboard
        """
        obj_points, img_points = [], []
        img_shape = None

        # Object points for all the images is the same.
        # The object points are only in the xy plane thous z axes is zero.
        img_obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), dtype=np.float32)
        img_obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        print('Calibrating camera')

        # Run on all images in the calibration folder.
        pbar = tqdm(os.listdir(cal_images_path))
        for img_path in pbar:
            pbar.set_description('Processing {}'.format(img_path))
            img = mpimg.imread(os.path.join(cal_images_path, img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if found:
                img_points.append(corners)
                obj_points.append(img_obj_points)

                if img_shape is None:
                    img_shape = gray.shape

        pbar.close()

        if len(img_points) == 0:
            # If we got here we got a problem.
            raise ValueError("Didn't find chessboard pattern in any of the images in '{}'.".format(cal_images_path))

        print('Calculating calibration matrix and distortion coefficients.')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[::-1], None, None)
        self._camera_mtx = mtx
        self._camera_dist = dist
        print('Camera calibration done')

    def _create_bit_mask(self, src, low_tresh, high_tresh):
        """
        Creates a bit mask from `src` image when the bits are `1`
        only when `src` is between the thresholds.
        :param src: Source image
        :param low_tresh: Low threshold for mask
        :param high_tresh: High threshold for mask
        :return: A bit mask with the image shape.
        """
        bit_mask = np.zeros_like(src, dtype=np.bool)
        bit_mask[(low_tresh <= src) & (src <= high_tresh)] = 1
        return bit_mask

    def _get_edges(self, image, s_thresh, sobel_thresh, red_thresh):
        """
        Detects lane edges in an image via converting
        the image to HLS.
        The final result is created from the image S-channel and the
        xy derivative of the image L-channel.

        :param image: An image or a list of images
        :param s_thresh: A tuple or list representing the lower and upper
                         thresholds that will be used on the image S-channel
        :param sobel_thresh: A tuple or list representing the lower and upper
                             thresholds that will be being used on the xy
                             derivative of the image L-channel
        :return: A binary image or a list of binary images that
                 contains only the edges of the `images`
        """

        r_channel = image[:, :, 0]
        r_mask = self._create_bit_mask(r_channel, *red_thresh)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

#        l_channel = cv2.equalizeHist(l_channel)
        # Take the derivative of the l channel in both x and y directions
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
        # Calculate the magnitude using complex numbers math.
        sobel_mag = np.abs(sobel_x + 1j * sobel_y)
        # Scale the magnitude to the size of one byte.
        sobel_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))
        # Create bit mask
        sobel_mask = self._create_bit_mask(sobel_scaled, *sobel_thresh)

        s_mask = self._create_bit_mask(s_channel, *s_thresh)
        # Combine both masks using bitwise or
        combined_mask = s_mask | sobel_mask | r_mask
        return combined_mask

    def get_lane_view(self, image, s_thresh=(110, 255), sobel_thresh=(35, 60), red_thresh=(220, 255)):
        """
        Takes an image and process it using the following steps:
        * Fix camera distortions
        * Find edges in images
        * Warp to bird-eye view
        :param image: The image to be processed.
        :param s_thresh: Lower & Upper HLS S-Channel thresholds for edge detection.
        :param sobel_thresh: Lower & Upper Sobel derivative thresholds for edge detection.
        :param red_thresh: Lower & Upper RGB R-Channel thresholds for edge detection
        :return: A binary that contains the image lanes in bird-eye view and the undistorted image.
        """
        assert self._camera_mtx is not None, 'Camera must be calibrated first!'
        undistorted = cv2.undistort(image, self._camera_mtx, self._camera_dist)
        edges = self._get_edges(undistorted, s_thresh, sobel_thresh, red_thresh)
        warped = cv2.warpPerspective(np.float32(edges), self._wrap_mat, edges.shape[::-1], flags=cv2.INTER_LINEAR)
        return warped, undistorted

    def _draw_lane(self, image, lane_color,
                   left_line_color, left_line_points, left_line_poly_coeffs,
                   right_line_color, right_line_points, right_line_poly_coeffs):
        """
        Draws a lanes on an image
        :param image: An undistorted image
        :param lane_color: The lane color in RBG format
        :param left_line_color: The left line color in RBG format
        :param left_line_points: A tuple with y,x point of the lane for this specific image
        :param left_line_poly_coeffs: A list with the line polynomial coefficients.
        :param right_line_color: The left line color in RBG format
        :param right_line_points: A tuple with y,x point of the lane for this specific image
        :param right_line_poly_coeffs: A list with the line polynomial coefficients.
        :return:
        """
        # Create an image to draw the lines on
        binary_zero = np.zeros(image.shape[0:2]).astype(np.uint8)
        warped_overlay = np.dstack((binary_zero, binary_zero, binary_zero))

        # Draw the lane area on the overlay
        # Calculate lines
        plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fit_x = np.polyval(left_line_poly_coeffs, plot_y)
        right_fit_x = np.polyval(right_line_poly_coeffs, plot_y)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(warped_overlay, np.int_([pts]), lane_color)
        # Draw the lines on the overlay
        warped_overlay[left_line_points[0], left_line_points[1]] = left_line_color
        warped_overlay[right_line_points[0], right_line_points[1]] = right_line_color

        # Warp the overlay back to the image perspective
        overlay = cv2.warpPerspective(warped_overlay, self._wrap_mat_inv, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        result = cv2.addWeighted(image, 1, overlay, 0.6, 0)
        return result

    def _add_info(self, image, curvature, left_line_pos, right_line_pos):
        """
        Adds additional text info to the picture
        :param image: The image
        :param curvature: The curvature of the lane
        :param left_line_pos: The distance in meters of left line from center
        :param right_line_pos: The distance in meters of right line from center
        :return: The image with text
        """
        cv2.putText(image, 'Radius of Curvature = {}(m)'.format(np.int(curvature)), (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        # Find the shift from center
        arg_max = np.argmax((left_line_pos, right_line_pos))
        lane_width = left_line_pos + right_line_pos

        shift_text = 'Vehicle is {}m {} of center'
        if arg_max == 0:
            # The shift is to the left
            shift = np.round(left_line_pos - lane_width/2, 2)
            shift_text = shift_text.format(shift, 'left')
        else:
            # The shift is to the right
            shift = np.round(right_line_pos - lane_width/2, 2)
            shift_text = shift_text.format(shift, 'right')

        cv2.putText(image, shift_text.format(np.int(curvature)), (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        return image

    def draw_on_image(self, image, left_lane_line: Line, right_lane_line: Line):
        """
        Draws lane on `image` and add additional info at the top left of the image
        :param image: An undistorted image
        :param left_lane_line: The left lane line
        :param right_lane_line: The right lane line
        :return: An image with lane highlighted and lane data written on it
        """
        processed = self._draw_lane(image, (0, 255, 0),
                                    (255, 0, 0), left_lane_line.get_points(), left_lane_line.best_fit,
                                    (0, 0, 255), right_lane_line.get_points(), right_lane_line.best_fit)

        curvature = np.min((left_line.radius_of_curvature, right_line.radius_of_curvature))
        processed = self._add_info(processed, curvature, left_line.line_base_pos, right_line.line_base_pos)

        return processed


def detect_lane(binary, lines, movie_mode=True, max_skip=0):
    """
    Detects a line in a binary, this method detects one
    line only and searches for it only in the predefined
    `detection_zone`.
    This method will try tracking the line first when possible
    and on failure will try finding it by scanning the `detection_zone`.
    :param max_skip: The maximum number of frames the algorithm will avoid
                     finding the line if tracking is failed
    :param movie_mode: If this flag is `True` line tracking and skipping will
                       be enabled.
    :param lines: A tuple with both the lane lines
    :param binary: A binary.
    """

    # ########### Helper methods ####################################
    def validate_lane_lines(binary, left_fit, right_fit, error_margin=200):
        """
        Checks if the detected lines are valid.
        :param right_fit: The polynomial coefficients of the right line
        :param left_fit: The polynomial coefficients of the left line
        :param binary: A binary
        :param error_margin: The margin of error of the mean distance
                             between the lines
        :return: `True` if the lines are valid and `False` otherwise.
        """
        y_points = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
        x_left = np.polyval(left_fit, y_points)
        x_right = np.polyval(right_fit, y_points)

        # Check lines distance and that they don't meet
        distances = x_right - x_left
        if any(distances <= 0):
            print('validation failed, error: Crossing lines')
            return False
        normed_distances = np.abs(distances - np.mean(distances))
        if any(normed_distances >= error_margin):
            print('validation failed, error: {} is above error margin'.format(np.max(normed_distances)))
            return False

        return True

    def find_lane(binary, lines, n_windows=19, recenter_thresh=50, win_margin=100):
        """
        Looks for a line in a binary image.
        Scans the `detection_zone` in order to find
        the line using the following steps:
        * Calculate bottom half histogram in order
          to find the initial searching point
        * Use sliding windows in order to follow the
          line from the initial point and collect the points
          that make up the line along the way
        * Use the collected points in order to interpolate
          the line using numpy.polyfit with degree of 2.

        :param lines: A tuple containing the lane lines
        :param binary: A binary image.
        :param n_windows: The number of windows to use.
        :param recenter_thresh: The minimum number of pixels that need
                                to be found in order to justify recalculation
                                of window center.
        :param win_margin: The margin of the window in the x axes.
        """
        # Extract the lane lines
        left_line, right_line = lines

        midpoint = binary.shape[1]//2
        # Calculate the histogram of the bottom half of the image
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        # Find the peak location
        curr_pos_left = np.argmax(histogram[:midpoint])
        curr_pos_right = np.argmax(histogram[midpoint:]) + midpoint
        # Use sliding windows in order to find the line
        window_height = np.int(binary.shape[0] / n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Create list for the line pixel indices
        left_line_idx = []
        right_line_idx = []

        for window in range(n_windows):
            # Identify window boundaries in x and y
            win_y_high = binary.shape[0] - window * window_height
            win_y_low = win_y_high - window_height
            win_x_left_low = curr_pos_left - win_margin
            win_x_left_high = curr_pos_left + win_margin
            win_x_right_low = curr_pos_right - win_margin
            win_x_right_high = curr_pos_right + win_margin
            # Identify the nonzero pixels in x and y within the window
            nonzero_win_left_idx = ((nonzero_y >= win_y_low) &
                                    (nonzero_y <= win_y_high) &
                                    (nonzero_x >= win_x_left_low) &
                                    (nonzero_x <= win_x_left_high)).nonzero()[0]
            nonzero_win_right_idx = ((nonzero_y >= win_y_low) &
                                     (nonzero_y <= win_y_high) &
                                     (nonzero_x >= win_x_right_low) &
                                     (nonzero_x <= win_x_right_high)).nonzero()[0]
            # Add points indices to their corresponding list
            left_line_idx.append(nonzero_win_left_idx)
            right_line_idx.append(nonzero_win_right_idx)
            # Recenter the window if needed
            if len(nonzero_win_left_idx) > recenter_thresh:
                # Recenter to the indices mean position
                curr_pos_left = np.mean(nonzero_x[nonzero_win_left_idx], dtype=np.int)
            if len(nonzero_win_right_idx) > recenter_thresh:
                # Recenter to the indices mean position
                curr_pos_right = np.mean(nonzero_x[nonzero_win_right_idx], dtype=np.int)

        left_line_idx = np.concatenate(left_line_idx)
        right_line_idx = np.concatenate(right_line_idx)

        # If we didn't find both of the lines then the detection failed.
        if not any(left_line_idx) or not any(right_line_idx):
            left_line.detected = right_line.detected = False
            return

        left_line_x = nonzero_x[left_line_idx]
        left_line_y = nonzero_y[left_line_idx]
        left_line_fit = np.polyfit(left_line_y, left_line_x, 2)

        right_line_x = nonzero_x[right_line_idx]
        right_line_y = nonzero_y[right_line_idx]
        right_line_fit = np.polyfit(right_line_y, right_line_x, 2)

        if not validate_lane_lines(binary, left_line_fit, right_line_fit):
            return

        if left_line.current_fit is not None:
            left_line.diffs = left_line.current_fit - left_line_fit
            right_line.diffs = right_line.current_fit - right_line_fit

        left_line.current_fit = left_line_fit
        right_line.current_fit = right_line_fit

        left_line.all_x, left_line.all_y = left_line_x, left_line_y
        right_line.all_x, right_line.all_y = right_line_x, right_line_y

        left_line.detected = right_line.detected = True

    def track_lane(binary, lines, margin=20):
        """
        Tracks the lines based on latest fit found
        :param margin: The margin in which to track the lines in.
        :param lines: A tuple containing the lane lines
        :param binary: A binary
        :return: `True` if the lines were found, and `False` otherwise.
        """
        # Extract lane lines
        left_line, right_line = lines
        # If the line was not detected last frame then there is nothing to track
        if (not left_line.detected) or (not left_line.detected):
            return False

        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_fit_val_x = np.polyval(left_line.current_fit, nonzero_y)
        right_fit_val_x = np.polyval(right_line.current_fit, nonzero_y)

        left_line_idx = ((nonzero_x > (left_fit_val_x - margin)) & (nonzero_x < (left_fit_val_x + margin)))
        right_line_idx = ((nonzero_x > (right_fit_val_x - margin)) & (nonzero_x < (right_fit_val_x + margin)))

        # If we didn't find both of the lines then the detection failed.
        if not any(left_line_idx) or not any(right_line_idx):
            left_line.detected = right_line.detected = False
            return False

        left_line_x = nonzero_x[left_line_idx]
        left_line_y = nonzero_y[left_line_idx]
        left_line_fit = np.polyfit(left_line_y, left_line_x, 2)

        right_line_x = nonzero_x[right_line_idx]
        right_line_y = nonzero_y[right_line_idx]
        right_line_fit = np.polyfit(right_line_y, right_line_x, 2)

        if not validate_lane_lines(binary, left_line_fit, right_line_fit):
            return False

        left_line.diffs = left_line.current_fit - left_line_fit
        right_line.diffs = right_line.current_fit - right_line_fit

        left_line.current_fit = left_line_fit
        right_line.current_fit = right_line_fit

        left_line.all_x, left_line.all_y = left_line_x, left_line_y
        right_line.all_x, right_line.all_y = right_line_x, right_line_y

        return True

    # ###############################################################

    # Crop the binary to the `detection_zone`
    left_line, right_line = lines
    global skip_count

    # Try to track the lanes first (only in movie mode) and if tracking fails,
    # decide if to use previous fit info or to find the lane again.
    if not movie_mode or not track_lane(binary, lines):
        if movie_mode and left_line.best_fit is not None and skip_count < max_skip:
            skip_count += 1
        else:
            find_lane(binary, lines)

    if left_line.detected:
        skip_count = 0


def average_lines(binary, lines, alpha=0.8):
    """
    Calculate the best fit of the lines via averaging the fitted_x points via
    weighted mean, by the following formula, best_x=(1-alpha)*best_x + alpha*current_fit_x
    :param binary: The binary
    :param lines: A tuple with both left & right lane lines
    :param alpha: The alpha number of the weighted mean
    :return:
    """
    left_line, right_line = lines

    # Calculate the x point for each of the lines
    y_points = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    x_left = np.polyval(left_line.current_fit, y_points)
    x_right = np.polyval(right_line.current_fit, y_points)

    # # Calculate the best x
    # left_line.best_x = np.average(left_line.recent_fitted_x, axis=0)
    # right_line.best_x = np.average(right_line.recent_fitted_x, axis=0)

    left_line.best_x = left_line.best_x*(1-alpha) + x_left*alpha
    right_line.best_x = right_line.best_x*(1-alpha) + x_right*alpha

    # Calculate the best fit
    left_line.best_fit = np.polyfit(y_points, left_line.best_x, 2)
    right_line.best_fit = np.polyfit(y_points, right_line.best_x, 2)


def update_curvature_and_pos(binary, lines):
    """
    Updates curvature and position from center
    """
    # Extract lane lines
    left_line, right_line = lines
    # Calculate meter per pixel (mpp) in both axis
    mpp_y = LANE_HEIGHT_METERS / binary.shape[0]
    mpp_x = LANE_WIDTH_METERS / LANE_WIDTH_PIXELS

    y_eval = binary.shape[0] * mpp_y

    # Fit new polynomials to x,y in world space
    left_line_y, left_line_x = left_line.all_y, left_line.all_x
    left_line_fit = np.polyfit(left_line_y * mpp_y, left_line_x * mpp_x, 2)

    right_line_y, right_line_x = right_line.all_y, right_line.all_x
    right_line_fit = np.polyfit(right_line_y * mpp_y, right_line_x * mpp_x, 2)

    # Calculate curvature at the bottom of the image
    # For left line
    first_der = [2*left_line_fit[0], left_line_fit[1]]
    second_der = 2*left_line_fit[0]

    numerator = (1 + np.polyval(first_der, y_eval)**2)**1.5
    denominator = np.abs(second_der)
    left_line.radius_of_curvature = numerator / denominator

    # For right line
    first_der = [2*right_line_fit[0], right_line_fit[1]]
    second_der = 2*right_line_fit[0]

    numerator = (1 + np.polyval(first_der, y_eval)**2)**1.5
    denominator = np.abs(second_der)
    right_line.radius_of_curvature = numerator / denominator

    # Calculate position
    center_pos = (binary.shape[1] / 2) * mpp_x
    left_lane_pos = np.polyval(left_line_fit, y_eval)
    right_lane_pos = np.polyval(right_line_fit, y_eval)

    left_line.line_base_pos = np.abs(left_lane_pos - center_pos)
    right_line.line_base_pos = np.abs(right_lane_pos - center_pos)


def process_image(image, movie_mode=True):
    """
    Advanced lane line processor, assumes that
    :param movie_mode: Enables/Disables the following:
                       * lane tracking (use previous fit to find the current fit).
                       * skip detection when tracking fails (use the same fit as the last one
                         for `max_skip` number of times).
                       * calculate the best fit by weighted average.
    :param image: input image
    :return: processed image
    """
    lanes, undist = camera.get_lane_view(image)
    lines = (left_line, right_line)

    detect_lane(lanes, lines, movie_mode=movie_mode, max_skip=4)

    if left_line.detected and right_line.detected:
        if movie_mode:
            average_lines(lanes, lines)
        else:
            left_line.best_fit = left_line.current_fit
            right_line.best_fit = right_line.current_fit

        update_curvature_and_pos(lanes, lines)

    # If there is no fit to use - leave the image as is.
    if left_line.best_fit is not None:
        return camera.draw_on_image(undist, left_line, right_line)

    return undist


if __name__ == '__main__':
    output_folder = 'output_images'
    input_path = 'test_images'
    input_path = 'project_video.mp4'

    print("Starting lane detection pipeline. input={} output={}".format(input_path, output_folder))

    camera = Camera()
    # Calibrate the camera
    camera.calibrate_camera('camera_cal', (9, 6))

    # Global parameters
    left_line = Line()
    right_line = Line()
    # set `skip_count` to np.inf to make sure that on movie mode
    # the first frame won't be skipped.
    skip_count = np.inf

    print('Processing...')
    # Create files list
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
    else:
        files = [input_path]

    for file in files:
        if os.path.isdir(input_path):
            file_path = os.path.join(input_path, file)
        else:
            file_path = input_path

        if not os.path.isfile(file_path):
            continue

        suffix = file.split('.')[1]
        if suffix == 'jpg':
            # Image processing pipeline
            img = mpimg.imread(file_path)
            dst = process_image(img, movie_mode=False)
            print(file)
            mpimg.imsave(os.path.join(output_folder, file), dst)
        elif suffix == 'mp4':
            # Video processing pipeline
            clip = VideoFileClip(file_path)
            dst = clip.fl_image(process_image)
            dst.write_videofile(os.path.join(output_folder, file), audio=False)

    print('Done')
