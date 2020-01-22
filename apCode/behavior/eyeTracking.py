# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:02:41 2019.

@author: pujalaa
"""
import numpy as np
import apCode.volTools as volt
from dask import delayed, compute

def cropImagesAroundFishWithU(images, unet, cropSize=None, downSample=None,\
                              verbose=1):
    """ Given a set of images and a trained U net, uses the network to detect
    fish (assuming 1 fish/image) in the images and to crop to the specified 
    size around the fish.
    """

    imgDims = images.shape[-2:]
    if cropSize is None:
        cropSize = unet.input_shape[1:3]
    images_rs = volt.img.resize(images,unet.input_shape[1:3])
    images_pred = np.squeeze(unet.predict(images_rs[...,None]))
    if not downSample is None:
        imgDims_new = np.round(np.array(imgDims)*downSample).astype(int)
        images = volt.img.resize(images,imgDims_new)
    imgDims = images.shape[-2:]
    images_pred_rs = volt.img.resize(images_pred,imgDims)
    images_crop = images*images_pred_rs
    cent = compute(*[delayed(volt.img.findCentroid)(img) for img in images_crop])  
    images_crop = volt.img.cropImgsAroundPoints(images, cent, cropSize=cropSize)
    return images_crop
