import numpy as np
from itertools import combinations

import skimage
import skimage.exposure

import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\zebrafish-analysis')
from image_process import brightest_portion, take_biggest_blob

def distance(pts):
    ((x1, y1), (x2, y2)) = pts
    return np.hypot(x2-x1, y2-y1)

def closest_blobs(props):
    centroids = map(lambda x: x.centroid, props)
    combination_indices = list(combinations(range(len(props)), r=2))
    distances = list(map(distance, combinations(centroids, r=2)))
    closest_pair_index = np.argmin(np.array(distances))
    return tuple(map(lambda x: props[x], (combination_indices[closest_pair_index])))

def just_eyes(binary_image):
    new_image = np.zeros_like(binary_image)
    labeled = skimage.measure.label(binary_image)
    props = skimage.measure.regionprops(label_image=labeled, intensity_image=binary_image, cache=True)
    nontiny_blobs = list(filter(lambda x: x.area > 2, props))
    if len(nontiny_blobs) < 2:
        return new_image # skimage.transform.resize(skimage.data.horse(), (150, 150), mode='reflect')
    (blob1, blob2) = closest_blobs(nontiny_blobs)
    for [r, c] in blob1.coords:
        new_image[r, c] = True
    for [r, c] in blob2.coords:
        new_image[r, c] = True
    return new_image

def get_blobs(binary_image):
    integered = skimage.img_as_int(binary_image)
    labeled = skimage.measure.label(integered)
    props = skimage.measure.regionprops(label_image=labeled, intensity_image=integered, cache=True)
    return props

I_crop = np.load('all-cropped-images.pickle')
I_crop_scaled = np.array(list(map(skimage.exposure.rescale_intensity, I_crop)))
I_crop_dynamic_threshold = np.array(list(map(lambda img: brightest_portion(img, 0.001), I_crop_scaled)))
I_crop_dynamic_int = np.array(list(map(skimage.img_as_int, I_crop_dynamic_threshold)))
I_crop_just_eyes = np.array(list(map(just_eyes, I_crop_dynamic_int)))
