#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:50:23 2017

@author: Nikhil S Hubballi
"""

import cv2
import argparse
from features import FeatureMatching

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-one", "--image_one", required=True,	help="path to the first image")
ap.add_argument("-two", "--image_two", required=True,	help="path to the second image")
args = vars(ap.parse_args())

# Load the images
imageA = cv2.imread(args["image_one"])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
imageB = cv2.imread(args["image_two"])

# Stitch the images
stitch = FeatureMatching()

# Step 1: Apply Histogram equilization
imageA = stitch.equalizeHist(imageA)
imageB = stitch.equalizeHist(imageB)

# Step 2: Extract Key Points and Features from the image
kpA, featuresA = stitch.detectAndComputeKP(imageA)
kpB, featuresB = stitch.detectAndComputeKP(imageB)

#Step 3: Compare the features, identify the Good Matches and find Homography Matrix
matchesVerified, Val, Mask = stitch.matchKeyPoints(kpA, kpB, featuresA, featuresB)

# Step 4: Stitch Images together
stitched = stitch.stitchImage(imageB, imageA, Val)

# Creates an image showing the features matched between both the images
showMatch = stitch.drawMatches(imageA, imageB, kpA, kpB, matchesVerified)

#Show and Save the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Feature Matches", showMatch)
cv2.imwrite('Show_Match.jpg',showMatch)
cv2.imshow("Stitched", stitched)
cv2.imwrite('Stitched.jpg',stitched)
cv2.waitKey(0)
cv2.destroyAllWindows() 


