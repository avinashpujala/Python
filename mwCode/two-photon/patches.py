import numpy as np

def make_patches(image, box_radius):
    """
    Divide an image into a matrix of patches.

    This method divides an image into a series of overlapping patches with a
    specified radius. This allows operations to easily be mapped over the
    spatial dimensions of an image. The resulting array is smaller in the first
    two dimensions, because it cannot generate patches for the pixels less than
    or equal to box_radius pixels from the edge.
    """
    boxSize = 2 * box_radius + 1

    loRow = box_radius
    hiRow = image.shape[0] - box_radius
    loCol = box_radius
    hiCol = image.shape[1] - box_radius

    centerRows = list(range(loRow, hiRow))
    centerCols = list(range(loCol, hiCol))

    boxes = np.empty((len(centerRows), len(centerCols), boxSize, boxSize), dtype=image.dtype)
    for r in centerRows:
        for c in centerCols:
            box = image[r-box_radius:r+box_radius+1, c-box_radius:c+box_radius+1]
            boxes[r - box_radius, c - box_radius] = box
    return boxes
