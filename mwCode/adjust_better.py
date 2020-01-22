import skimage
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/woottenm/Documents/Code/zebrafish-analysis/')
from snippets.assorted import map_np
from crop_circle import *

def get_mean_image(image_stack):
    return None

image_list = []
for i in range(2, 751):
    image_list.append(skimage.io.imread('C:/Users/woottenm/Pictures/Zebrafish/f1_{:06d}.bmp'.format(i)))
I = np.array(image_list)
del image_list

mean_image = skimage.exposure.rescale_intensity(I.mean(axis=0))

within_dish_indices = get_circle_coords(mean_image, radius_adjust=-20)
mean_image[within_dish_indices] = mean_image[within_dish_indices].mean()

I_adjusted = I - mean_image

del I
del mean_image

I_adjusted_scaled = map_np(skimage.exposure.rescale_intensity, I_adjusted)
del I_adjusted

for i in range(2, 751):
    skimage.io.imsave('no-ghost-{:03d}.png'.format(i), I_adjusted_scaled[i])
