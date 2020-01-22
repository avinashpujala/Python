- Annotate frames
  - Open the frame stack you're trying to annotate in ImageJ (File -> Import -> Import Sequence)
  - Crop the frames to a reasonable size (right around the eyes)
  - Save the images as a TIF stack
  - Scale the images up 5x with order 0. This might not be strictly necessary,
    but the low resolution annotations look really blocky, and make you
    second-guess your annotations.
  - Use the freehand selection tool to outline the eyes. Use Control-T to add
    them to the ROI manager
  - When done, select all the ROIs and save them as a zip file. It's really easy
    to accidentally save just one if you're not paying attention (speaking from
    experience)
- Train classifier
  - The methods have individual documentation, but they work in concert like so:
```
edges = classifier_train.read_annotated_edges('all-rois.zip')
patches = classifier_train.get_patches(imread('cropped-images.tif'))
(pos, neg) = classifier_train.categorize_patches(patches, edges)
pos_expand = classifier_train.expand_without_noise(pos, len(neg))
X, Y = classifier_train.combine_classes(pos_expand, neg)
clf = classifier_train.train_classifier(X, Y)
```
- Actually compute the orientations
  - For this, the input images actually have to be cropped so that only one eye
    is in the image
  - Make sure that there is always a three-pixel margin around the eyes. Because
    how this classifier works, it will *never* classify a pixel in that zone as
    an edge.
  - These images don't have to be aligned with the training images in any way.
```
pmap = classifier_run.get_probability_map(topEyes[1], clf)
classifier_run.get_orientation(pmap)
```
