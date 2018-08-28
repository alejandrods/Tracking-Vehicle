#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:20:51 2017

@author: Alex
"""

import numpy as np
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip


##############
## 0. DATA
##############

images = glob.glob('*vehicles/*/*')

cars = []
notcars = []

for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)
        
print(len(cars))
print(len(notcars))



##############
## 1. INFORMATION OF DATA
##############

def data_info(car_list, notcar_list):
    print('Number of car images:', len(car_list))

    print('Number of notcar images:', len(notcar_list))

    example_img = mpimg.imread(car_list[0])
    print('Shape of images: ', example_img.shape)
    
    return

data_info(cars, notcars)



##############
## 2. SHOW RANDOM IMAGES
##############

index = np.random.randint(1, len(cars))
example_cars = mpimg.imread(cars[index])
example_notcars = mpimg.imread(notcars[index])

plt.figure()
plt.imshow(example_cars)
plt.title('Cars')

plt.figure()
plt.imshow(example_notcars)
plt.title('Notcars')



##############
## 3. GET HOG FEATURES
##############

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    
#TEST WITH 1 IMAGE OF CAR
# Read in the image
gray = cv2.cvtColor(example_cars, cv2.COLOR_RGB2GRAY)

# Define HOG parameters
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block

hog_features, hog_image = get_hog_features(gray, orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
plt.figure()
plt.imshow(hog_image, cmap='gray')
plt.title('HOG image')



##############
## 4. GET SPATIAL FEATURES
##############

def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

#TEST WITH 1 IMAGE OF CAR
spatial_features = bin_spatial(example_cars)

plt.figure()
plt.plot(spatial_features)
plt.title('Spatial Features')



##############
## 5. GET HISTOGRAM FEATURES
##############

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    hist1 = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    hist2 = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    hist3 = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    
    return hist_features

#TEST WITH 1 IMAGE OF CAR
hist_features = color_hist(example_cars)

plt.figure()
plt.plot(hist_features)
plt.title('Histogram Features')



##############
## 6. EXTRACT FEATURES FOR ALL IMAGES
##############

#THIS IS USED TO GET FEATURES FOR ONLY 1 IMAGE
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

#THIS IS USED TO GET FEATURES FOR DATASET IMAGE
#source: https://sites.coecis.cornell.edu/chaowang/2017/03/10/self-driving-car-vehicle-detection-and-tracking/
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     nbins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=nbins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

#PARAMETERS
color_space = 'YCrCb' # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
spatial_size = (32, 32) 
nbins = 64
hog_channel = 'ALL' # 0, 1, 2, or "ALL"
spatial_feat = True 
hist_feat = True 
hog_feat = True 
slide_window_sizes = [80, 96, 112, 128, 160] 
slide_window_overlap = (0.60, 0.60)  
y_start_stop = [400, 600] 

#Extract features
cars_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, nbins=nbins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat,
                            hog_feat=hog_feat)

notcars_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, nbins=nbins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat,
                            hog_feat=hog_feat)

print('Length of car features',len(cars_features))
print('Length of notcar features',len(notcars_features))



##############
## 7. TRAIN AND TEST MODEL
##############

#NORMALIZE FEATURES
#Create array stack
X = np.vstack((cars_features, notcars_features)).astype(np.float64)     
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, 
                                                    random_state=rand_state)

print('Training Data Set:', X_train.shape)
print('Test Data Set:', X_test.shape)

#TRAIN AND TEST MODEL. SAVE MODEL
#Linear SVC 
svc = LinearSVC()
svc.fit(X_train, y_train)

#Save model
joblib.dump((svc, X_scaler), 'model_1.pkl')



##############
## 8. CREATE SLIDE WINDOW AND BOXES
##############

#SLIDE WINDOW
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

#SEARCH WINDOW
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

#source: https://github.com/Dalaska/CarND-P15-Vehicle-Detection-and-Tracking/blob/master/Vehicle_Detection.ipynb
def search_windows_sizes(image, slide_window_sizes,
                    clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):
    hot_windows = []
    for xywin in slide_window_sizes:
        xy_window = (xywin, xywin)

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=xy_window, xy_overlap=slide_window_overlap)

        hot = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        hot_windows.extend(hot)
    return hot_windows

#source: https://github.com/hfoffani/Vehicle-Detection/blob/master/vehicle-detection.ipynb
def draw_labeled_bboxes(img, labels, color=(0, 0, 255), thickness = 8):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thickness)
    # Return the image
    return img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

#To remove false positives
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



##############
## 9. TEST IMAGE
##############
    
#Plot Test
for img_file in glob.glob('test_images/' + '*.jpg'):

    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # normalize.
    image = image.astype(np.float32)/255
        
    hot_windows = search_windows_sizes(image, slide_window_sizes,
                             svc, X_scaler, color_space=color_space,
                             spatial_size=spatial_size, hist_bins=nbins,
                             orient=orient, pix_per_cell=pix_per_cell,
                             cell_per_block=cell_per_block,
                             hog_channel=hog_channel, spatial_feat=spatial_feat,
                             hist_feat=hist_feat, hog_feat=hog_feat)
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels, color=(255, 215, 80))

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.subplot(122)
    plt.imshow(draw_img)
    plt.title('Cars')
    plt.show()
    
    
    
##############
## 9. FINAL FUNCTION
##############
    
save_heat = []
def process_image(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    image = image_rgb.astype(np.float32)/255
    

    hot_windows = search_windows_sizes(image, slide_window_sizes,
                            svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=nbins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    # will accumulate between frames!
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    
    #Save heat
    save_heat.append(heat)
    heat = np.sum(save_heat, axis=0)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1 * len(save_heat))

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    return draw_labeled_bboxes(np.copy(img), labels, color=(255, 215, 80))


#VIDEOS
#1. Test_video
#white_output = 'Test_final.mp4'
#clip1 = VideoFileClip('test_video.mp4')
#white_clip = clip1.fl_image(process_image)
#white_clip.write_videofile(white_output, audio=False)

#2. Project_video
white_output = 'Project_video_FINAL.mp4'
clip1 = VideoFileClip('project_video.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
        


