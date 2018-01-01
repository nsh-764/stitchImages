# Stitch Images

This is a program to stitch two images

The program uses ORB descriptor of opencv and BruteForce Matcher to identify keypoints and features between two images.Then calulating the Homography from the features, stitches two images and creates a mosaic image.

version 1.0.0


# Dependencies Required

1.Numpy

2.opencv

3.argparse

# Database configuration

Use the imageA and imageB to be stitched in any of the following formats

1. Windows bitmaps - *.bmp, *.dib

2. JPEG files - *.jpeg, *.jpg, *.jpe

3. JPEG 2000 files - *.jp2

4. Portable Network Graphics - *.png

5. WebP - *.webp

6. Portable image format - *.pbm, *.pgm, *.ppm

7. Sun rasters - *.sr, *.ras

8. TIFF files - *.tiff, *.tif


# How to run the program :

In the terminal, execute:

python stitch.py -one "PATH_TO_FIRST_IMAGE" -two "PATH_TO_SECOND_IMAGE"

or

python stitch.py --image_one "PATH_TO_FIRST_IMAGE" --image_two "PATH_TO_SECOND_IMAGE"

# Output:

showMatch.jpg - An image showing the features that are matched between the two input images

stitched.jpg - The resulting mosaic from the stitching

# Author: 

Nikhil S Hubballi

nikhil.hubballi@gmail.com

