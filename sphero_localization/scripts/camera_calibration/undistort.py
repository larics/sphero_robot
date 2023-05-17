"""
This script can be used to validate the camera calibration results. It reads the
saved calibration images from files, applies the calibrated camera model, and
saves the new undistored images to disk.
"""
import glob
import json
from pathlib import Path

import cv2 as cv
import numpy as np


## Make your changes here...
calibration_filename = 'camera_south.json'
############################

# Load calibration parameters.
path = Path(__file__).parent / '../../config' / calibration_filename
with open(path, 'r') as json_file:
    camera_data = json.load(json_file)
dist = np.array(camera_data["dist"])
mtx = np.array(camera_data["mtx"])

# Load images
load_path = Path(__file__).parent / 'pictures/raw'
images = list(load_path.glob('calibrate*.png'))
print(len(images), "images found")

assert len(images) > 0

frame = cv.imread(images[0])
h, w = frame.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (h, w), 0, (h, w))
x, y, w1, h1 = roi
yh1 = y + h1
xw1 = x + w1

save_path = Path(__file__).parent / 'pictures/fixed'

for fname in images:
    img = cv.imread(fname)

    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # dst = dst[y:yh1, x:xw1]

    cv.imshow('img', dst)
    cv.imwrite(f"{save_path}/remapped_{fname.partition('_')[2]}", dst)
    cv.waitKey(1000)