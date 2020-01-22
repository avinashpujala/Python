import numpy as np
import scipy.interpolate

import skimage
import skimage.exposure
import skimage.transform
import skimage.morphology

import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\zebrafish-analysis')
from image_process import brightest_portion, remove_tiny_blobs

def blur(image, sigma=3):
    return skimage.filters.gaussian(image, sigma)

def width_along_line(image, x0, y0, x1, y1):
    mask = np.zeros_like(image)
    rr, cc = skimage.draw.line(y0, x0, y1, x1)
    mask[rr, cc] = True
    return np.logical_and(~image, mask).sum()

def width_profile_from_axis(image, axis_coords, length=10):
    xs, ys = (axis_coords[:, 0], axis_coords[:, 1])
    xs_spline = scipy.interpolate.make_interp_spline(range(len(xs)), xs, 3)
    ys_spline = scipy.interpolate.make_interp_spline(range(len(ys)), ys, 3)
    def width(t):
        dx_dt = xs_spline.derivative()(t)
        dy_dt = ys_spline.derivative()(t)
        norm  = (dx_dt ** 2) + (dy_dt ** 2)
        Δx = dx_dt / norm * length
        Δy = dy_dt / norm * length
        x0 = int(xs[t] + Δy)
        x1 = int(xs[t] - Δy)
        y0 = int(ys[t] - Δx)
        y1 = int(ys[t] + Δx)
        return width_along_line(image, x0, y0, x1, y1)

    return np.array(list(map(width, range(len(axis_coords)))))

I_crop = np.load('all-cropped-images.pickle')
I_crop_scaled = np.array(list(map(skimage.exposure.rescale_intensity, I_crop)))
I_crop_dynamic_threshold = np.array(list(map(lambda x: brightest_portion(x, fraction=0.025), I_crop_scaled)))
I_binary_clean = np.array(list(map(remove_tiny_blobs, I_crop_dynamic_threshold)))
I_blur = np.array(list(map(blur, I_binary_clean)))
I_blur_binary = np.array(list(map(lambda x: brightest_portion(x, fraction=0.98), I_blur)))
I_thin = np.array(list(map(skimage.morphology.thin, I_blur)))
