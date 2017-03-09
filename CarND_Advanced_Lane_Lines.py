
# coding: utf-8

# In[20]:


import glob
import matplotlib.pyplot as plt
from collections import deque
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import numpy as np
import cv2


class AdvancedLaneFinding():
    
    
        
    ###### Global variables #############
    # Useful to indicate first run (meaning blind search and no prev lines)
    firstRun = True
    # Indicate reset search (blind search again)
    resetSearch = False
    # To check if lines detected are "good"
    isGoodDetection = True
        
    # Queues to store last 10 detectec lines
    colaR = deque(maxlen=10)
    colaL = deque(maxlen=10)
        
    # wrong in a row
    wrongConsecutive=0
        
    # Variables that will be used as instance of Line class
    prevLines,newLines = None, None
    
    # For camera calibration
    ret,mtx, dist,rvecs,tvecs = 0,0,0,0,0
    
    def __init__(self):
        

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = 0,0,0,0,0
        
        ###### Global variables #############
        # Useful to indicate first run (meaning blind search and no prev lines)
        self.firstRun = True
        # Indicate reset search (blind search again)
        self.resetSearch = False
        # To check if lines detected are "good"
        self.isGoodDetection = True
        
        # Queues to store last 10 detectec lines
        self.colaR = deque(maxlen=10)
        self.colaL = deque(maxlen=10)
        
        # wrong in a row
        self.wrongConsecutive=0
        
        # Variables that will be used as instance of Line class
        self.prevLines, self.newLines = None, None
        
    #This method calibrates the camera based on a classic example of a chessboard. This is an easy and well proven way
    # to calibrate a camera.
    def calibrateCamera(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
    
        # Load an image to perform calibration
        img = cv2.imread('./camera_cal/calibration2.jpg')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    
        # Add object points, image points
        objpoints.append(objp)
        imgpoints.append(corners)
    
        # Draw and display the corners
        ##img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
                
        return ret, mtx, dist, rvecs, tvecs
        
        
        
    def prepareLaneDetection(self):
        
       self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrateCamera()
    
    # In[21]:
    
    # This function performs color thresolding using HLS color space. 
    # Specifically, the thresholding is applied on H channel
    def threshold_hls(self, img, thresh=(0, 255)):
        threshMin = thresh[0]
        threshMax = thresh[1]
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        H = hls[:,:,2]
        binary = np.zeros_like(H)
        binary[(H > threshMin) & (H <= threshMax)] = 1
        # 3) Return a binary image of threshold result
        binary_output = binary 
        return binary_output
    
    
    # In[22]:
    
    # This function performs absolute sobel thresolding using HLS color space.
    # It performs threshold X gradient on L channel and threshold color channel on S channel.
    # The result of both is combined ant the one-channel picture is returned.
    def abs_sobel_thresh_hls(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        
        # Applying Sobel on X direction
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
        # Return combined binary
        return combined_binary
    
    
    # In[23]:
    
    # Function that performs X or Y Sobel treshold on a color RGB picture
    def abs_sobel_thresh(self,img, orient='x', thresh_min=0, thresh_max=255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))# Take the derivative in x
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))# Take the derivative in y
        
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
        # Return the result
        return binary_output
    
    
    # In[24]:
    

    
    # Performs a blind search creating a histogram to detec peaks which would indicate lane lines detected
    def blindSearch(self, binary_warped, visualize = False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 110
        # Set minimum number of pixels found to recenter window
        minpix = 90
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
    
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        if visualize == True:
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
        
        return left_fit, right_fit, left_fitx, right_fitx, ploty
    
    
    # In[25]:
    
    def lineFinding(self,binary_warped, left_fit, right_fit, visualize=False):    
        # from the next frame of video (called "binary_warped") we can detect the lane lines based on 
        # fitted values (left_fit, right_fit).
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a margin for the window where it searchs for lines detected
        margin = 60
        # Gets the indexes of pixels where lines were detected
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if sum(leftx)< 500 or sum(rightx)<500:
            return None, None, None, None, None
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
        #And you're done!. We can also visualize the result here as well
    
        if visualize == True:
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            
        return left_fit, right_fit, left_fitx, right_fitx, ploty
    
    
    # In[26]:
    
    
    
    def calculateCurvature(self, left_fitx, right_fitx, ploty, plot=False):
        leftx = left_fitx
        rightx = right_fitx
        # Fit a second order polynomial to pixel positions in each lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
      
        #Now we have polynomial fits and we can calculate the radius of curvature as follows:
    
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        #print(left_curverad, right_curverad)
        # Example values: 1926.74 1908.48
    
        #But now we need to stop and think... We've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So we actually need to repeat this calculation after converting our x and y values to real world space.
        #This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.
        #So here's a way to repeat the calculation of radius of curvature after correcting for scale in x and y:
    
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/600 # meters per pixel in y dimension
        xm_per_pix = 3.7/540 # meters per pixel in x dimension
    
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
        return left_curverad, right_curverad
    
    
    # In[27]:
    
    def plotLines(self, warped, original, linesDetected):
        
        # Extract needed variables from Line instance
        left_fitx, right_fitx = linesDetected.left_fitx, linesDetected.right_fitx
        left_fit, right_fit = linesDetected.left_fit, linesDetected.right_fit
        Minv, ploty = linesDetected.Minv, linesDetected.ploty
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Based on lines already detected, search for lane lines with a smaller margin, in order to eliminate the noise
        # This is not really necesary, it is done just to draw the lane lines as limits 
        margin = 40
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))    
        
        # Based on lane detected, gets point for the lane lines
        ptsL = np.hstack((np.array([np.transpose(np.vstack([left_fitx-15, ploty]))]),  np.array([np.flipud(np.transpose(np.vstack([left_fitx+15, ploty])))])) )
        ptsR = np.hstack((np.array([np.transpose(np.vstack([right_fitx-15, ploty]))]),  np.array([np.flipud(np.transpose(np.vstack([right_fitx+15, ploty])))])) )
        
        #Plot color lane lines
        color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.fillPoly(color_warp, np.int_([ptsL]), (255,0, 0))
        cv2.fillPoly(color_warp, np.int_([ptsR]), (0,0, 255))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)
        
        return result
    
    
    # In[28]:
    
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    
    # In[34]:
    
    def undistort_warp(self, img, mtx, dist):
        
        # Undistort using mtx and dist
        img = cv2.undistort(img, mtx, dist, None, mtx)
        
        # Apply different image filters in order to get the most posible clean picture of the lane lines.
        # Different filters are used, each of them detecting some particular pixels that will be joined later
        # in a single picture. For more detail of each filter, please refer to each called method.
        
        # Result of filter threshold_hls (binary threshold on H channel)
        result_flt  = self.threshold_hls(img, thresh=(110, 255))
        # Result of filter abs_sobel_thresh_hls (binary sobel threshold on L,S channel)
        result_flt2 = self.abs_sobel_thresh_hls(img,s_thresh=(170, 255), sx_thresh=(30,200))
        # Result of filter abs_sobel_thresh(binary sobel threshold on RGB channel, Y direction)
        result_flt3 = self.abs_sobel_thresh(img, orient='y', thresh_min=100, thresh_max=150)
        # Result of filter abs_sobel_thresh(binary sobel threshold on RGB channel, X direction)
        result_flt4 = self.abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=255)
        
        # Create a black picture where we will put the pixels detecte by previous filters
        combined_binary = np.zeros_like(result_flt)
        # Turning on pixels by combining contours detected by the filters
        combined_binary[(result_flt == 1) | (result_flt2 == 1) | (result_flt3 == 1) | (result_flt4 == 1)] = 1
    
        # Get picture's shape
        imshape = combined_binary.shape
        
        #Define vertices for the region of interest
        vertices = np.array([[(150,imshape[0]),(540, 440), (770, 440), (imshape[1]-90,imshape[0])]], dtype=np.int32)
        #vertices = np.array([[(150,imshape[0]),(600, 440), (720, 440), (imshape[1]-150,imshape[0])]], dtype=np.int32)
         
        # Remove unneeded information from picture by choosing a region of interest that will only keep the road
        combined_binary = self.region_of_interest(combined_binary, vertices)
        
        # Define 4 source points for perspective transformation
        src = np.float32([[480, 510], [800, 510], [80,720], [1200,720]])
        dst = np.float32([[330, 510], [985, 510], [300,720], [1000, 720]])
        
        #Compute the perspective transform:
        M = cv2.getPerspectiveTransform(src, dst)
        #Compute the inverse perspective transform:
        Minv = cv2.getPerspectiveTransform(dst, src)
        #Warp the image using the perspective transform, M:
        warped = cv2.warpPerspective(combined_binary, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        
        return warped, M, Minv
    
    
    # In[35]:
    
    
    
    # Class created to keep track of detected lines
    class Line():
        
        def __init__(self, left_fit, right_fit, left_fitx, right_fitx, ploty, Minv, imgShape ):
            #Inverse perspective
            self.Minv = Minv
            # was the line detected in the last iteration?
            self.detected = False
            # lines detected in search
            self.left_fit = left_fit
            self.right_fit = right_fit
            # lines created based on search detection
            self.left_fitx = left_fitx
            self.right_fitx = right_fitx
            # Y values for lines
            self.ploty = ploty
            # radius of curvature per line
            self.left_curverad = None
            self.right_curverad = None
            # Image shape
            self.imgShape = imgShape
            
        # Calculate distance to the center of the road
        def getLineBasePos(self):
            xm_per_pix = 3.7/540 # meters per pixel in x dimension    
            camera_center = (self.left_fitx[-1] + self.right_fitx[-1])/2 #Assumes camera is perfectly centered
            center_diff = (camera_center - self.imgShape[1]/2)*xm_per_pix #Calculates based on lines detected
            return center_diff
        
        # Checks if the lines are separated by at least 3.7 m.
        def isGoodSeparation(self):
            xm_per_pix = 3.7/540 # meters per pixel in x dimension    
            lines_distance = ( np.max(self.right_fitx) - np.min(self.left_fitx))*xm_per_pix #Calculates based on lines detected
            if lines_distance < 3.7:
                return False
            else:
                return True
            
        # Average the over the last 10 well detected iterations so that lines are smoother
        def setLineAverage(self, colaL, colaR, reset):
            colaL.append(self.left_fit)
            colaR.append(self.right_fit)
            # Averages only is there is more than one line already detected
            if len(colaL)>1:
                self.left_fit, self.right_fit = np.mean(colaL, axis=0), np.mean(colaR, axis=0)
    
    
    # In[36]:

    
    # Checks if lines detected make sense in comparison to previous good lines
    def goodLines(self,prevLine, newDetected):
        # Calculates the sum of squares of the difference of x fits
        s = np.sum((prevLine.left_fitx-newDetected.left_fitx)**2)
        s2 = np.sum((prevLine.left_fitx-newDetected.left_fitx)**2)
        # maxium difference allowed
        sqrtSumMax = 300000
        # If any of the lines has a greater difference, it means that lines are not very similar to previous ones
        if (s > sqrtSumMax):
            return False
        if (s2 > sqrtSumMax):
            return False
        else:
            return True
    
    
    # In[40]:
    
    
    def processLanePipeline(self, img):
        
        # Start by getting an undistorted and warped binary picture where the lane lines are visible and ready to 
        # be detected. Also saves perspective transformation and it's inverse. 
        undistorted, perspective_M, Minv = self.undistort_warp(img, self.mtx, self.dist)
        #self.firstRun, self.prevLines, self.newLines, self.colaL, self.colaR, self.isGoodDetection, self.wrongConsecutive, self.resetSearch
        
        # Check if it is first run 
        if self.firstRun == True or self.resetSearch == True:
            # If first run, then uses blind search to detec the lines.
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.blindSearch(undistorted, visualize=False)
            self.prevLines = self.Line(left_fit, right_fit, left_fitx, right_fitx, ploty, Minv, img.shape) 
            # Calculates radius of curvature for detected lines
            left_curverad, right_curverad = self.calculateCurvature(self.prevLines.left_fitx, self.prevLines.right_fitx, self.prevLines.ploty, plot=False)
            self.prevLines.left_curverad, self.prevLines.right_curverad = left_curverad, right_curverad
            
            if self.resetSearch == True:
                self.colaL.clear()
                self.colaR.clear()
            
            self.firstRun = False
            self.resetSearch = False
            self.wrongConsecutive = 0
            
        else:
            # If not the first run, it can use the previous lines to detec the new ones.
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.lineFinding(undistorted, self.prevLines.left_fit, self.prevLines.right_fit, visualize=False) 
            if left_fit == None:
                self.wrongConsecutive += 1
                left_fit, right_fit, left_fitx, right_fitx, ploty = self.prevLines.left_fit, self.prevLines.right_fit, self.prevLines.left_fitx, self.prevLines.right_fitx, self.prevLines.ploty
            
            # Calculates radius of curvature for detected lines
            left_curverad, right_curverad = self.calculateCurvature(left_fitx, right_fitx, ploty, plot=False)
       
        # if last detection was not good, remove the values from the queue
        # so that they don't mess the average
        if  self.isGoodDetection == False:
            self.wrongConsecutive += 1
            if len(self.colaR)>0:
                self.colaR.pop()
                self.colaL.pop()
            else:
                self.resetSearch=True
        else:
            self.wrongConsecutive = 0
            
        if self.wrongConsecutive == 3:
            self.resetSearch = True
        
        
        # Create an instance with new lines detected and add curve radius
        self.newLines = self.Line(left_fit, right_fit, left_fitx, right_fitx, ploty, Minv, img.shape)
        self.newLines.left_curverad, self.newLines.right_curverad = left_curverad, right_curverad 
        
        # Are the lines correctly detected in comparison to previous lines?
        isGoodDetection = self.goodLines(self.prevLines, self.newLines)
        if self.newLines.isGoodSeparation() == False:
                self.resetSearch = True
                print("no good separation")
        
        # Averages lines coordinates 
        self.newLines.setLineAverage(self.colaL, self.colaR, False)
        
        self.prevLines = self.newLines
        
        # Gets final picture whit lane plotted
        finalPicture = self.plotLines(undistorted, img, self.newLines)
        # Message if a frame had bad detection
        error=''
        if isGoodDetection == False:
            error = 'Bad detection'
            print(error)
        
        # Print in the frame the radius of curvature and distance to the center
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(finalPicture, error, (200, 20), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(finalPicture, 'Radius of Curvature: '+ str(round(self.newLines.right_curverad, 3)) + ' m', (200, 50), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(finalPicture, 'Center Distance: '+ str(round(self.newLines.getLineBasePos(),2)) + ' m', (200, 75), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return finalPicture
    
    
