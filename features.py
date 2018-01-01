#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:48:49 2017

@author: Nikhil S Hubballi
"""

# import the necessary packages
import numpy as np
import cv2

class FeatureMatching:
    def equalizeHist(self, image):
        """
        When local intensity distributions differ significantly at tile edges,
        a noticeable shift in appearance may remain after mosaicing the images.
        Histogram Equilization is done in order to adjust the intensities of
        the images by enhancing the contrast.

        input: image in BGR format

        output: image in BGR format

        """
        # Convert the image first into YUV from BGR and apply histogram euilization
        imageYUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        imageYUV[:,:,0] = cv2.equalizeHist(imageYUV[:,:,0])

        # Convert the image back into BGR
        image = cv2.cvtColor(imageYUV, cv2.COLOR_YUV2BGR)

        # Return the euilized BGR format image
        return image
    
    
    def detectAndComputeKP(self, image):
        """
        For stitching two images, it's essential to identify the keypoints and
        features between the two images. ORB (Oriented FAST and Rotated BRIEF)
        descriptor of the opencv is used for the job.

        input: image in BGR format

        output: kps (keypoints) in object format
                features in numpy array format

        """
        # Using ORB (Oriented FAST and Rotated BRIEF) descriptors
        # to identify key points and descriptors

        detector = cv2.ORB_create()
        kps, features = detector.detectAndCompute(image, None)

        # Return keypoint objects and features numpy array
        return kps, features
    

    def matchKeyPoints(self, kpA, kpB, featuresA, featuresB):
        """
        After identifying the keypoints and features in both the images to be
        stitched, features are to be matched. Brute-Force Matcher from the
        opencv is used for this process. It takes the descriptor of one feature
        in first set and is matched with all other features in second set using
        some distance calculation. After identifying the raw matches, only good
        mathces within a certain ratio of distances are taken. Homography Matrix
        is computed for these identified good matches.

        input: imageA, imageB
               
        output: matchesVerified
                Val - Homography matrix
                Mask


        """
        verify_ratio=0.8
        reprojThresh=0.5
        
        # Compute the raw matches using Brute Force Matcher object
        # and initialise for verified matches

        matcher = cv2.BFMatcher()
        matchesRAW = matcher.knnMatch(featuresA,featuresB, k=2)
        matchesVerified = []

        # Identify the good matches by comparing the distance with verify_ratio
        for m,n in matchesRAW:
            if m.distance < verify_ratio * n.distance:
                matchesVerified.append(m)

        # Computing Homography with a minimum match count
        MIN_MATCH_COUNT = 8

        if len(matchesVerified) > MIN_MATCH_COUNT:
            kPointsA = []
            kPointsB = []

            # Add the matching points to an array
            for match in matchesVerified:
                kPointsA.append(kpA[match.queryIdx].pt)
                kPointsB.append(kpB[match.trainIdx].pt)

            kPointsA = np.float32(kPointsA).reshape(-1,1,2)
            kPointsB = np.float32(kPointsB).reshape(-1,1,2)

            # Compute the Homography between the two
            Val, Mask = cv2.findHomography(kPointsA, kPointsB, cv2.RANSAC, reprojThresh)

            # Return the verified matches along with computed Homography matrix and mask
            return matchesVerified, Val, Mask
        else:
            exit()


    def drawMatches(self, imageA, imageB, kpA, kpB, matchesVerified):
        """
        This is to visualize the features matched between the two images.

        input: imageA, imageB - both the images in BGR format
               kpA, kpB - keypoints identified in both the images
               matchesVerified - Identified matches

        output: Image (data type: uint8)

        """
        # generate output image
        (heightA, widthA) = imageA.shape[:2]
        (heightB, widthB) = imageB.shape[:2]
        visualize = np.zeros((max(heightA, heightB), widthA + widthB, 3), dtype="uint8")

        imageCompare = cv2.drawMatches(imageA, kpA, imageB, kpB, matchesVerified ,visualize, flags=2)

        # Return an image showing matched keypoints between the images
        return imageCompare

    def stitchImage(self, img1, img2, M):
        """
        After finding the Homography matrix for the images, we need to apply 
        perspective warp to the images on a new canvas to obtain the stitched 
        image.
        
        input: img1 - imageB
               img2 - image1
               
        output: stitched - stitched image
        
        """
        # Get the dimensions of input images
        w1,h1 = img1.shape[:2]
        w2,h2 = img2.shape[:2]

        # Calculate the dimensions for Canvas
        img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
        img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

        # Calculate relative perspective of second image using 
        # Homography matrix - M
        img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

        # Resulting dimension of the stitched Image
        result_dims = np.concatenate((img1_dims, img2_dims), axis = 0)

        # Stitch Images together
        # Calculate the dimensions of match points
        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        # Create output array after Affine transformation
        transform_dist = [-x_min,-y_min]
        transform_array = np.array([[1, 0, transform_dist[0]],
                                    [0, 1, transform_dist[1]],
                                    [0,0,1]])
        
        # Warp images to get the stitched image
        stitched = cv2.warpPerspective(img2, transform_array.dot(M),(x_max-x_min, y_max-y_min))
        stitched[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = img1

        # Return the stitched image
        return stitched
