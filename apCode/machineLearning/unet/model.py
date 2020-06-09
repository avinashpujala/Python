
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K


# Metric function
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, gamma=2):
    """
    Based on the following paper. Addresses class imbalance by downweighting
    the loss contribution from easy examples (y_pred >> 0.5)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2018). Focal
    Loss for Dense Object Detection. ArXiv:1708.02002 [Cs].
    """
    alpha=0.25 # wt for positive class
    p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
    alpha_t = y_true*alpha + (1-y_true)*(1-alpha)
    loss = -alpha_t*((1-p_t)**gamma)*K.log(p_t+1e-3)
    return loss

def get_unet(img_width=512, img_height=512, img_channels=1, activation='elu',
             kernel_initializer='he_normal', optimizer='adam',
             loss='binary_crossentropy'):
    inputs = Input((img_height, img_width, img_channels))
    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coef])
    return model



def get_unet_parallel(img_width=512, img_height=512, img_channels=1,
                      activation='elu', kernel_initializer='he_normal',
                      optimizer='adam',
                      loss='binary_crossentropy'):
    from keras.utils import multi_gpu_model
    import tensorflow as tf
    import keras.backend.tensorflow_backend as tfback
    def _get_available_gpus():
        """Get a list of available gpu devices (formatted as strings).

        # Returns
            A list of available GPU devices.
        """


        if tfback._LOCAL_DEVICES is None:
            devices = tf.config.list_logical_devices()
            tfback._LOCAL_DEVICES = [x.name for x in devices]
        return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
    tfback._get_available_gpus = _get_available_gpus
    inputs = Input((img_height, img_width, img_channels))
    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation=activation,
                kernel_initializer=kernel_initializer, padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    parallel_model = multi_gpu_model(model)

    parallel_model.compile(optimizer=optimizer, loss=loss,
                           metrics=[dice_coef])
    return parallel_model

#def mask_to_rle(preds_test_upsampled):
#    """
#    Iterate over the test IDs and generate run-length encodings for each
#    separate mask identified by skimage
#    """
#    new_test_ids = []
#    rles = []
#    for n, id_ in enumerate(test_ids):
#        rle = list(prob_to_rles(preds_test_upsampled[n]))
#        rles.extend(rle)
#        new_test_ids.extend([id_] * len(rle))
#    return new_test_ids,rles

def prob_to_rles(x, cutoff=0.5):
    from skimage.morphology import label
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def rle_encoding(x):
    """
    Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    """
    import numpy as np
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
