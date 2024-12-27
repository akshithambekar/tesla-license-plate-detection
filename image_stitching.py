import cv2 as cv
import numpy as np
from imutils import paths
import argparse
import imutils

# input: /Users/akshithambekar/Documents/Code/input_folder
# output: /Users/akshithambekar/Documents/Code/output_folder

ap = argparse.ArgumentParser()
ap.add_argument('--i', '--images', type=str, required=True, help='input images directory path')
ap.add_argument('--o', '--output', type=str, required=True, help='output image directory path')
args = vars(ap.parse_args())

print('loading images')
imgPaths = sorted(list(paths.list_images(args['i'])))
images = []
for imgPath in imgPaths:
    image = cv.imread(imgPath)
    images.append(image)
stitcher = cv.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if status == 0:
    cv.imwrite(args['o'], stitched)
    cv.imshow('Stitched', stitched)
    cv.waitKey(0)
else:
    print('image stitching failed')