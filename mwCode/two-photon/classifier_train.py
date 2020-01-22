import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from imageio import mimwrite
from tqdm import tqdm_notebook, tnrange
from itertools import product as cartesian_product
import random
import joblib
import read_roi
from patches import make_patches
from scipy.ndimage import binary_fill_holes
from sklearn.ensemble import RandomForestClassifier

from typing import Dict, Tuple, Set, List
Coordinate = Tuple[np.int32, np.int32]

def roi_to_coords(roi):
    xs = np.array(roi['x'])
    ys = np.array(roi['y'])
    arr = (np.concatenate([[xs, ys]]) / 5).round().astype(int).T
    return set(map(tuple, arr))

def read_annotated_edges(annotation_file: str, scale_factor=5) -> Dict[int, Set[Coordinate]]:
    """
    Reads an ImageJ ROI stack into a convenient data structure.

    Parameters
    ----------
    annotation_file : str
        Path to the file containing the ROIs
    scale_factor : int (optional, default = 5)
        If the annotations correspond to a scaled-up image, the scale factor
        used to annotate those images. This is primarily for convenience, as the
        final coordinates are integers anyways.

    Returns
    -------
    scaled_rois : Dict[int, Set[Coordinate]]
        All the coordinates of the ROIs drawn, indexed by frame. If there is
        more than one ROI in a frame, the one that ends up in the image is
        arbitrary.
    """
    rois = read_roi.read_roi_zip(annotation_file)
    scaled_rois = dict((roi['position'] - 1, roi_to_coords(roi)) for roi in rois.values())
    return scaled_rois

def get_patches(image_stack):
    """
    Extracts the patches from a stack of images

    Parameters
    ----------
    image_stack : ndarray with shape (time, width, height)
        A stack of images to use for the training dataset.

    Returns
    -------
    patches : ndarray with shape (time, (width - 6), (height - 6), 7, 7)
    """
    return np.array(list(map(lambda img: make_patches(img, 3), image_stack)))

def categorize_patches(patches: np.ndarray, coordinates: Dict[int, Set[Coordinate]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Make edge annotations into positive and negative examples

    Parameters
    ----------
    patches: ndarray
        see output of get_patches
    coordinates: Dict[int, Set[Coordinate]]
        see output of read_annotated_edges

    Returns
    -------
    pos: List[np.ndarray]
        Positive training patches, unraveled to a single dimension
    neg: List[np.ndarray]:
        Negative training patches, unraveled to a single dimension
    """
    boxSize = patches.shape[3]
    boxRadius = int((boxSize - 1) / 2)
    centerRows = list(range(boxRadius, boxRadius + patches.shape[1]))
    centerCols = list(range(boxRadius, boxRadius + patches.shape[2]))
    pos = []
    neg = []
    for t in coordinates.keys():
        for (r, c) in cartesian_product(centerRows, centerCols):
                appropriateClass = pos if (c, r) in coordinates[t] else neg
                appropriateClass.append(patches[t, r-boxRadius, c-boxRadius].ravel())
    return (pos, neg)

def expand_without_noise(existing_data, target_length):
    """
    Through random repetition, expand a dataset to the specified size
    """
    additional_data = target_length - len(existing_data)
    new_data = np.array(random.choices(existing_data, k=additional_data))
    return np.concatenate([existing_data, new_data])

def combine_classes(positive_examples, negative_examples):
    """
    Transform the (positive, negative) input style into the (X, Y) input style
    needed by scikit-learn and other ML libraries
    """
    shuffled_indices = list(range(len(positive_examples) + len(negative_examples)))
    random.shuffle(shuffled_indices)
    X = np.concatenate([negative_examples, positive_examples])
    Y = np.array([0] * len(negative_examples) + [1] * len(positive_examples))
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]
    return (X, Y)

def train_classifier(X, Y):
    """
    Train a classifier mapping X onto Y

    This uses a random forest with 100 trees. All other parameters follow
    the defaults of scikit-learn.
    """
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X, Y)
    return clf
