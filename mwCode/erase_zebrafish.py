import numpy as np

from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from skimage.io import imread, imsave
from image_process import take_biggest_blob

import sys
sys.path.append('C:/Users/woottenm/Documents/zebrafish-analysis/')
from snippets.assorted import map_np
from crop_circle import *

def erase_zebrafish(mean_cropped, thresh=0.7):
    new_mean = np.copy(mean_cropped)
    binary = new_mean < thresh
    labeled = label(binary)
    rp = sorted(regionprops(labeled), key=lambda x: x.area, reverse=True)
    (x, y) = tuple(map(int, rp[1].centroid))
    # I may have botched this...
    cropped_small = new_mean[x-100:x+100, y-100:y+100]
    canny_biggest_small = take_biggest_blob(canny(cropped_small))
    hull = convex_hull_image(canny_biggest_small)
    mean_not_hull = cropped_small[~hull].mean()
    cropped_small[hull] = mean_not_hull
    return new_mean

if __name__ == '__main__':
    I = []
    for i in range(2, 751):
        I.append(skimage.io.imread('V:/Matthew Wootten/midlineDetectionTesting_ellipses/f1_{:06d}.bmp'.format(i)))
    I = np.array(I)
    mean_image = skimage.exposure.rescale_intensity(I.mean(axis=0))
    cropped = crop_circle(mean_image, radius_adjust=-20)
    background = erase_zebrafish(cropped)
    I_adjusted = I - background
    I_scaled = map_np(skimage.exposure.rescale_intensity, I_adjusted)
    for i in range(len(I_scaled)):
        skimage.io.imsave('erased-{:03d}.png'.format(i), I_scaled[i])
