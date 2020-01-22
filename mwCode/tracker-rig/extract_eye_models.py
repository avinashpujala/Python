"""
General usage:

I = the full image stack, with dimensions time * length * width

Code ::

    global_thresh = extract_eye_models.intermodes(I[0])
    binary_imgs = I < global_thresh
    eye1 = []
    eye2 = []
    for binary_img in binary_imgs:
        models = extract_eye_models.eye_models(binary_img)
        eye1.append(models[0])
        eye2.append(models[1])
    eyeA, eyeB = extract_eye_models.correlate(eyes1, eyes2)
    eyeA_orients = np.array(eyesA)[:, 4]
    eyeB_orients = np.array(eyesB)[:, 4]
    Δθ = extract_eye_models.angle_difference(eyeA_orients, eyeB_orients)
"""

import numpy as np
from skimage.measure import regionprops, label

def eye_models(binary_image):
    """
    Return ellipses corresponding to each eye.

    Output format: two-element array of tuples, each containing information
    about a single eye

    Order of parameters:
    - x-coordinate (image column)
    - y-coordinate (image row)
    - Major axis length
    - Minor axis length
    - Orientation
    """
    blobs = regionprops(label(binary_image))
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    firstFive = blobs[0:5]
    firstFew = list(filter(lambda b: b.eccentricity < 0.95 and 500 < b.area < 2000, firstFive))
    (eye1, eye2) = min(zip(firstFew, firstFew[1:]), key=lambda tup: abs(tup[0].area - tup[1].area))
    return [eye_model_from_blob(eye1), eye_model_from_blob(eye2)]

def eye_model_from_blob(blob):
    x = blob.centroid[1]
    y = blob.centroid[0]
    a = blob.major_axis_length
    b = blob.minor_axis_length
    θ = blob.orientation
    return (x, y, a, b, θ)

def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.hypot(dx, dy)

def correlate(eye1, eye2):
    """
    Set up correspondences between eyes.

    Since eye1 and eye2 are determined by only relative size, they may not
    always be the same eye from frame to frame. This method tries to split these
    images into eyeA and eyeB, where eyeA and eyeB consistently track the same
    physical eye.
    """

    assert len(eye1) == len(eye2), "Eyes must come in pairs"

    # Start off this way; since A and B have no particular meaning, this could
    # just as well be reversed.
    eyeA = [eye1[0]]
    eyeB = [eye2[0]]

    # Skip frame 0, since it has already been assigned
    for i in range(1, len(eye1)):
        da1 = distance(eyeA[-1], eye1[i])
        da2 = distance(eyeA[-1], eye2[i])
        db1 = distance(eyeB[-1], eye1[i])
        db2 = distance(eyeB[-1], eye2[i])
        if (da1 < db1) and (db2 < da2):
            eyeA.append(eye1[i])
            eyeB.append(eye2[i])
        elif (da2 < db2) and (db1 < da1):
            eyeA.append(eye2[i])
            eyeB.append(eye1[i])
        else:
            raise RuntimeError('Ambiguous eye assignment')
    return (eyeA, eyeB)

def angle_difference(θ1, θ2):
    """
    Get the difference between two angles measured in radians, taking into
    account that angles must be subtracted in a special manner.
    """
    ordinary_diff = (θ2 - θ1) % np.pi
    return (np.pi / 2) - np.abs(ordinary_diff - (np.pi / 2))

def bimodtest(y):
    modes = 0
    for k in range(1, len(y) - 1):
        if y[k - 1] < y[k] > y[k + 1]:
            modes += 1
        if modes > 2:
            return False
    return modes == 2

def intermodes(I):
        """
        Find a global threshold for a grayscale image by choosing the threshold
        to be the mean of the two peaks of the bimodal histogram.

        Will fail and return 0 if the histogram is not bimodal
        """
        n = 255
        y = np.histogram(I.ravel(), bins=n+1, range=(0, n))[0].astype(float)
        iter = 0
        while not bimodtest(y):
            h = np.ones(3) / 3
            y = np.convolve(y, h, mode='same')
            iter += 1
            if iter > 10000:
                return 0

        TT = []
        for k in range(1, n - 1):
            if y[k - 1] < y[k] > y[k + 1]:
                TT.append(k)
        return int(np.floor(np.mean(TT)))
