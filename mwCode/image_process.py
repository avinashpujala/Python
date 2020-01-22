import skimage
import skimage.util
import skimage.measure
import skimage.morphology
import numpy as np

def brightest_portion(image, fraction):
    """
    Take (approximately) the brightest ``fraction`` proportion of the pixels in
    an image.

    This method works by taking an index into the sorted representation of the
    image. As a result, if the image has a large number of pixels set to the
    same value, then the

    Parameters
    ----------
    image : ndarray image
    """
    inv = skimage.util.invert(image)
    cutoff = np.sort(inv.ravel())[int((1-fraction) * image.size)]
    return inv >= cutoff

def remove_tiny_blobs(image, tiny_threshold=20):
    return skimage.morphology.remove_small_objects(image, min_size=tiny_threshold)

def crop_to_nonzero(image, border=2):
    (ys, xs) = image.nonzero()
    return image[min(ys)-border:max(ys)+border, min(xs)-border:max(xs)+border]

def take_biggest_blob(image):
    new_image = np.ones_like(image)
    inverted = skimage.util.invert(image)
    labeled = skimage.measure.label(inverted)
    blobs = skimage.measure.regionprops(labeled, cache=True)
    blobs_by_area = sorted(blobs, key=lambda blob: blob.area, reverse=True)
    biggest_blob = blobs_by_area[0]
    for [r, c] in biggest_blob.coords:
        new_image[r, c] = False
    return new_image

def get_blobs(binary_image, connectivity=2):
    integered = skimage.img_as_int(binary_image)
    labeled = skimage.measure.label(integered, connectivity=connectivity)
    props = skimage.measure.regionprops(label_image=labeled, intensity_image=integered, cache=True)
    return props

def filter_blobs(image, blob_predicate, connectivity=2):
    new_image = np.zeros_like(image)
    blobs = get_blobs(image, connectivity=connectivity)
    selected_blobs = filter(blob_predicate, blobs)
    for selected_blob in selected_blobs:
        for [r, c] in selected_blob.coords:
            new_image[r, c] = True
    return new_image
