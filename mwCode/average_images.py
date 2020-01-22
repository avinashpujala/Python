import sys
sys.path.append(r'C:\Users\woottenm\Documents\Code\avinash-code\Python\code')

import skimage
import skimage.io
import skimage.exposure
import numpy as np
import warnings
import apCode.volTools
from pathlib import Path

input_folder = sys.argv[1]
output_folder = sys.argv[2]
image_start = int(sys.argv[3])
image_end = int(sys.argv[4])
average_type = {
    'median': np.median,
    'mean': np.mean
}[sys.argv[5]]

images = apCode.volTools.img.readImagesInDir(inputFolder)

average_image = average_type(images, axis=0)
adjusted_images = images - average_image

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for i in range(image_start, image_end):
        name = Path(outputFolder) / 'adjusted-{:03d}.png'.format(i)
        skimage.io.imsave(name, skimage.exposure.rescale_intensity(adjusted_images[i - 50]))
