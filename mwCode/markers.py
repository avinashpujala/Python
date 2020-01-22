import numpy as np
from skimage import img_as_float
from skimage.exposure import rescale_intensity
import skimage.draw

def mark_with_circle(image : np.ndarray, r: np.float64, c: np.float64, color: str) -> np.ndarray:
    rgb_lookup = {
        'red':   np.array([255, 0, 0]),
        'green': np.array([0, 255, 0]),
        'blue':  np.array([0, 0, 255])
    }
    new_image = np.copy(image)
    rr, cc = circle(r, c, radius=2)
    new_image[rr, cc] = rgb_lookup[color]
    return new_image

def mark_two(img0, x1, y1, x2, y2):
    img1 = mark_with_circle(img0, y1, x1, 'red')
    img2 = mark_with_circle(img1, y2, x2, 'blue')
    return img2

def get_major_axis_line(center, orientation, length=2):
    (cy, cx) = center
    (dx, dy) = (length * np.cos(orientation), length * np.sin(orientation))
    xs = [cx + dx, cx - dx]
    ys = [cy - dy, cy + dy]
    return (xs, ys)

def draw_extension(annotate_image, center, orientation, length=10, fill=0.5):
    """
    Draws lines extending from a center of a length and orientation.

    Parameters
    ----------
    annotate_image : ndarray image
        The image to annotate
    center : array-like, length = 2
        The center of the line, in (row, column) format; will be converted to
        integer values
    orientation : float
        The orientation of the line, in radians. In some format.
    length : float (optional, default = 10)
        The length of the line on each side
    fill : whatever the image dtype is (optional, default = 0.5)
        The value to which each line pixel is set

    Returns
    -------
    A copy of the original image with the appropriate line drawn.
    """
    scaled = np.copy(rescale_intensity(img_as_float(annotate_image)))
    cs = get_major_axis_line(tuple(center), orientation, length=length)
    ([x0, x1], [y0, y1]) = cs
    line = skimage.draw.line(int(y0), int(x0), int(y1), int(x1))
    for (r, c) in zip(*line):
        try:
            scaled[r, c] = fill
        except IndexError:
            pass
    return scaled

def extend_blobs(binary_image, annotate_image, length=10):
    new_image = np.copy(rescale_intensity(img_as_float(annotate_image)))
    for blob in get_blobs(binary_image):
        new_image = draw_extension(new_image, blob.centroid, blob.orientation, length=length)
    return new_image
