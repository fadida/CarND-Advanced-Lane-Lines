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

LANE_HEIGHT_METERS = 30
LANE_WIDTH_METERS = 3.7

# The distance in pixels between lanes
LANE_WIDTH_PIXELS = 700


class Line:
    """
    This class handles line detection and tracking functionality
    """

    def __init__(self, detection_zone, n=1):
        """
        Initializes the line
        :param detection_zone: A tuple marking the start and end of
                                the region in the x axes.
        """
        self._detection_zone = detection_zone
        # was the line detected in the last iteration?
        self._detected = False
        self._n = n
        # x values of the last n fits of the line
        self._recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self._bestx = None
        # polynomial coefficients averaged over the last n iterations
        self._best_fit = None
        # polynomial coefficients for the most recent fit
        self._current_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self._diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self._allx = None
        # y values for detected line pixels
        self._ally = None

    def _find(self, binary, n_windows=9, recenter_thresh=50, win_margin=100):
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

        :param binary: A binary image.
        :param n_windows: The number of windows to use.
        :param recenter_thresh: The minimum number of pixels that need
                                to be found in order to justify recalculation
                                of window center.
        :param win_margin: The margin of the window in the x axes.
        """
        # Calculate the histogram of the bottom half of the image
        histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
        # Find the peak location
        peak_loc = np.argmax(histogram)

        # Use sliding windows in order to find the line
        window_height = np.int(binary.shape[0] / n_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        # Set current position to the peak location
        curr_pos = peak_loc
        # Create list for the line pixel indices
        line_idx = []

        for window in range(n_windows):
            # Identify window boundaries in x and y
            win_x_left = curr_pos - win_margin
            win_x_right = curr_pos + win_margin
            win_y_high = binary.shape[0] - window * window_height
            win_y_low = win_y_high - window_height
            # Identify the nonzero pixels in x and y within the window
            nonzero_win_idx = ((nonzero_y >= win_y_low) &
                               (nonzero_y <= win_y_high) &
                               (nonzero_x >= win_x_left) &
                               (nonzero_x <= win_x_right)).nonzero()[0]
            line_idx.append(nonzero_win_idx)
            # Recenter the window if needed
            if len(nonzero_win_idx) > recenter_thresh:
                # Recenter to the indices mean position
                curr_pos = np.mean(nonzero_x[nonzero_win_idx], dtype=np.int)

        line_idx = np.concatenate(line_idx)

        line_x = nonzero_x[line_idx] + self._detection_zone[0]
        line_y = nonzero_y[line_idx]

        line_fit = np.polyfit(line_y, line_x, 2)
        if self._current_fit is not None:
            self._diffs = self._current_fit - line_fit
        self._current_fit = line_fit

        self._allx = line_x
        self._ally = line_y

        self._detected = True

    def _track(self, binary, margin=100):
        """

        :param binary:
        :return:
        """
        # If the line was not detected last frame then there is nothing to track
        if not self._detected:
            return False

        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1]) + self._detection_zone[0]

        fit_val_x = np.polyval(self.get_poly(), nonzero_y)
        line_idx = ((nonzero_x > (fit_val_x - margin)) & (nonzero_x < (fit_val_x + margin)))

        line_x = nonzero_x[line_idx]
        line_y = nonzero_y[line_idx]

        if line_x.size == 0:
            self._detected = False
            return False

        line_fit = np.polyfit(line_y, line_x, 2)

        self._diffs = self._current_fit - line_fit
        self._current_fit = line_fit

        self._allx = line_x
        self._ally = line_y

        return True

    def detect(self, binary):
        """
        Detects a line in a binary, this method detects one
        line only and searches for it only in the predefined
        `detection_zone`.
        This method will try tracking the line first when possible
        and on failure will try finding it by scanning the `detection_zone`.
        :param binary: A binary.
        """
        # Crop the binary to the `detection_zone`
        left, right = self._detection_zone[0], self._detection_zone[1]
        cropped_binary = binary[:, left:right]
        if not self._track(cropped_binary):
            self._find(cropped_binary)

        self._update_properties()
        self._update_curvature_and_pos(binary)

    def get_points(self):
        """
        Returns the x,y points of the detected line
        """
        return self._ally, self._allx

    def get_poly(self):
        """
        Returns the polynomial coefficients of the line
        """
        return self._best_fit

    def _update_curvature_and_pos(self, binary):
        """
        Updates curvature and position from center
        """
        # Calculate meter per pixel (mpp) in both axis
        mpp_y = LANE_HEIGHT_METERS / binary.shape[0]
        mpp_x = LANE_WIDTH_METERS / LANE_WIDTH_PIXELS

        # Fit new polynomials to x,y in world space
        line_y, line_x = self.get_points()
        line_fit = np.polyfit(line_y * mpp_y, line_x * mpp_x, 2)

        # Calculate curvature at the bottom of the image
        first_der = [2*line_fit[0], line_fit[1]]
        second_der = 2*line_fit[0]
        y_eval = binary.shape[0] * mpp_y

        numerator = (1 + np.polyval(first_der, y_eval)**2)**1.5
        denominator = np.abs(second_der)
        self.radius_of_curvature = numerator / denominator

        # Calculate position
        center_pos = (binary.shape[1] / 2) * mpp_x
        lane_pos = np.polyval(line_fit, y_eval)
        self.line_base_pos = np.abs(lane_pos - center_pos)

    def _update_properties(self):
        n_samples = len(self._recent_xfitted)
        if n_samples == self._n:
            self._recent_xfitted.pop(0)
        else:
            n_samples += 1

        self._recent_xfitted.append(self._current_fit)

        # self._bestx = np.sum(self._recent_x, axis=0) / n_samples
        self._best_fit = np.sum(self._recent_xfitted, axis=0) / n_samples


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
        src_points = np.float32([[577, 462], [704, 462],
                                 [200, 720], [1082, 720]])
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

    def _get_edges(self, image, s_thresh, sobel_thresh):
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

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        s_mask = self._create_bit_mask(s_channel, *s_thresh)

        # Take the derivative of the l channel in both x and y directions
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
        # Calculate the magnitude using complex numbers math.
        sobel_mag = np.abs(sobel_x + 1j * sobel_y)
        # Scale the magnitude to the size of one byte.
        sobel_scaled = np.uint8(255 * sobel_mag / np.max(sobel_mag))
        # Create bit mask
        sobel_mask = self._create_bit_mask(sobel_scaled, *sobel_thresh)

        # Combine both masks using bitwise or
        combined_mask = s_mask | sobel_mask
        return combined_mask

    def get_lane_view(self, image, s_thresh=(130, 255), sobel_thresh=(90, 250)):
        """
        Takes an image and process it using the following steps:
        * Fix camera distortions
        * Find edges in images
        * Warp to bird-eye view
        :param image: The image to be processed.
        :param s_thresh: Lower & Upper S-Channel thresholds for edge detection.
        :param sobel_thresh: Lower & Upper Sobel derivative thresholds for edge detection.
        :return: A binary that contains the image lanes in bird-eye view and the undistorted image.
        """
        assert self._camera_mtx is not None, 'Camera must be calibrated first!'
        undistorted = cv2.undistort(image, self._camera_mtx, self._camera_dist)
        edges = self._get_edges(undistorted, s_thresh, sobel_thresh)
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
        result = cv2.addWeighted(image, 1, overlay, 0.3, 0)
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
                                    (255, 0, 0), left_lane_line.get_points(), left_lane_line.get_poly(),
                                    (0, 0, 255), right_lane_line.get_points(), right_lane_line.get_poly())

        curvature = np.min((left_line.radius_of_curvature, right_line.radius_of_curvature))
        processed = self._add_info(processed, curvature, left_line.line_base_pos, right_line.line_base_pos)

        return processed


def process_image(image):
    """
    Advanced lane line processor, assumes that
    :param image: input image
    :return: processed image
    """
    lanes, undist = camera.get_lane_view(image)

    left_line.detect(lanes)
    right_line.detect(lanes)
    # print(left_line.radius_of_curvature, right_line.radius_of_curvature)

    return camera.draw_on_image(undist, left_line, right_line)

if __name__ == '__main__':
    output_folder = 'output_images'
    input_path = 'test_images'
    #input_path = 'project_video.mp4'

    print("Starting lane detection pipeline. input={} output={}".format(input_path, output_folder))

    camera = Camera()
    # Calibrate the camera
    camera.calibrate_camera('camera_cal', (9, 6))

    img_width = 1280
    left_line = Line((0, img_width // 2))
    right_line = Line((img_width // 2, img_width))

    print('Processing...')
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
            left_line._detected = False
            right_line._detected = False
            img = mpimg.imread(file_path)
            dst = process_image(img)
            mpimg.imsave(os.path.join(output_folder, file), dst)
        elif suffix == 'mp4':
            left_line._n = 5
            right_line._n = 5
            clip = VideoFileClip(file_path)
            dst = clip.fl_image(process_image)
            dst.write_videofile(os.path.join(output_folder, file), audio=False)

    print('Done')
