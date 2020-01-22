import numpy as np

import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\zebrafish-analysis')
from using_cropped import *
from image_process import crop_to_nonzero

import skimage
import skimage.io
import skimage.draw
import skimage.color
import skimage.measure
import skimage.exposure
import skimage.morphology

def get_major_axis_line(blob, length=2):
    (cy, cx) = blob.centroid
    (dx, dy) = (length * np.cos(blob.orientation), length * np.sin(blob.orientation))
    xs = [cx + dx, cx - dx]
    ys = [cy - dy, cy + dy]
    return (xs, ys)

def extend_blobs(image, length=50):
    scaled = np.copy(skimage.exposure.rescale_intensity(skimage.img_as_float(image)))
    for blob in get_blobs(image):
        ([x0, x1], [y0, y1]) = get_major_axis_line(blob, length=length)
        line = skimage.draw.line(int(y0), int(x0), int(y1), int(x1))
        for (r, c) in zip(*line):
            try:
                scaled[r, c] = 0.5
            except IndexError:
                pass
    return scaled

I_with_lines = np.array(list(map(extend_blobs, I_crop_just_eyes)))

def angle_difference(θ1, θ2):
    ordinary_diff = (θ2 - θ1) % np.pi
    return (np.pi / 2) - np.abs(ordinary_diff - (np.pi / 2))

def angle_between_eyes(image):
    blobs = get_blobs(image)
    if len(blobs) != 2:
        return float('nan')
    [blob1, blob2] = blobs
    return angle_difference(blob1.orientation, blob2.orientation)

angle_diffs = np.array(list(map(angle_between_eyes, I_crop_just_eyes)))
