"""
Use Elastic Weight Consolidation to train the fish recognition model without
catastrophic forgetting

Usage
-----

- Make sure that the model file that defines the U-net is somewhere in
  PYTHONPATH
- If you have an existing network, use `add_fisher_to_existing_file` to compute
  the necessary Fisher information matrix
- Otherwise, use `train_model_from_scratch` to train a network and compute its
  Fisher information matrix together (this is somewhat faster than doing it
  seperately due to lower I/O overhead)
- When you want to add additional examples to a model, use
  `train_model_incrementally`

Implementation notes
--------------------

This follows [Kir17] fairly closely; however, it does differ in one notable way
when dealing with more than two training cycles. The original paper suggests
keeping track of all previous Fisher information matrices, while [Hus18]
advocates pooling all the penalty terms into one giant term, therefore needing
only a single Fisher information matrix. This implementation, for the sake of
simplicity, follows [Hus18].


References
----------
[Kir17]: Overcoming catastrophic forgetting in neural nets. James Kirkpatrick,
Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A.
Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis
Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell. Proceedings of the
National Academy of Sciences Mar 2017, 114 (13) 3521-3526; DOI:
10.1073/pnas.1611835114

[Hus18]: Note on the quadratic penalties in elastic weight consolidation. Ferenc
Huszár. Proceedings of the National Academy of Sciences Mar 2018, 115 (11)
E2496-E2497; DOI: 10.1073/pnas.1717042115

"""

import keras
import keras.backend as K
from keras import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import binary_crossentropy
import numpy as np
from copy import deepcopy
import h5py
from model import get_unet, dice_coef

def estimate_fisher_information(model, inputs, outputs, num_samples):
    """
    Estimate the Fisher information of parameters in the model. Note that this
    does not compute the whole matrix, as EWC only needs the diagonals of that
    matrix. As far as I can tell, the score is equivalent to the binary
    cross-entropy loss, and so this computes the average of the squares of the
    gradients of that loss for a sample of size `num_samples` from the old
    training set.
    """
    assert len(inputs) == len(outputs)
    random_indices = np.random.choice(len(inputs), num_samples)
    fisher_values = [np.zeros_like(weights) for weights in model.get_weights()]
    for random_index in random_indices:
        x = [inputs[random_index]]
        y = [outputs[random_index]]
        # Shamelessly stolen from https://github.com/keras-team/keras/issues/2226
        gradient_tensors = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        input_tensors = model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
        get_gradients = K.function(inputs=input_tensors, outputs=gradient_tensors)
        gradients = get_gradients([x, np.ones(len(x)), y, 1])
        for (layer_fisher, layer_grads) in zip(fisher_values, gradients):
            layer_fisher += (layer_grads ** 2)
    return [np.array(layer_fisher / num_samples) for layer_fisher in fisher_values]

def write_fisher(filename, fisher_values):
    with h5py.File(filename) as f:
        fisher_group = f.create_group('fisher_diagonal')
        for (i, values) in enumerate(fisher_values):
            fisher_group.create_dataset('values{:02d}'.format(i), data=values)

def train_model_from_scratch(X, Y, output_file, batch_size, epochs, fisher_samples=200):
    """
    Train a model from scratch, while saving the needed parts of the Fisher
    information matrix for later use in EWC

    X: inputs
        Should have 4 dimensions: num_samples * width * height * num_channels
    Y: outputs
        Should have 3 dimensions: num_samples * width * height
    output_file:
        The filename of an HDF5 file which will contain the weights and
        the approximate Fisher information matrix
    batch_size: integer
        The batch size for model training -- passed directly to the model
    epochs: integer
        The number of epochs for model training -- passed directly to the model
    fisher_samples: integer
        The number of samples to use for estimating the Fisher information matrix.
        This is the most time-consuming part of training, so reducing this value
        will do a lot to cut down on training time.
    """
    model = get_unet(X.shape[1], X.shape[2], X.shape[3])
    model.fit(X, Y, batch_size=batch_size, epochs=epochs)
    model.save(output_file)
    fisher = estimate_fisher_information(model, X, Y, num_samples=fisher_samples)
    write_fisher(output_file, fisher)

def train_model_incrementally(old_model_file, X, Y, output_file, batch_size=8, epochs=50, λ=100000000, fisher_samples=200):
    """
    Take a model which already includes an estimate of its Fisher information matrix
    and trains it more using EWC.

    old_model_file:
        the path to an HDF5 file containing weights in Keras format as well as the Fisher
        information matrix
    X:
        inputs for training
    Y:
        outputs for training
    output_file:
        The HDF5 file for output (again containing weights and Fisher information matrix, so
        this can be used in a chain)
    batch_size: integer
        The batch size for model training -- passed directly to the model
    epochs: integer
        The number of epochs for model training -- passed directly to the model
    λ: float
        The hyperparameter for EWC. Higher values make it stay closer to old weights, lower
        values let it optimize more for the current task. I've had good experience with values
        near the default of 10^8.
    fisher_samples: integer
        The number of samples to use for estimating the Fisher information matrix.
    """
    # Load the old model
    old_model = keras.models.load_model(old_model_file, custom_objects={'dice_coef': dice_coef})
    # Load the old Fisher diagonal
    fisher = []
    with h5py.File(old_model_file) as f:
        weight_keys = sorted(f['fisher_diagonal'].keys())
        for key in weight_keys:
            fisher.append(np.array(f['fisher_diagonal'][key]))
    # Create the new loss function
    def ewc_loss(model):
        def loss(y_true, y_pred):
            standard_loss = binary_crossentropy(y_true, y_pred)
            ewc_term = 0
            for layerIndex in range(len(fisher)):
                Δweights = model.trainable_weights[layerIndex] - old_model.get_weights()[layerIndex]
                ewc_term += K.sum((λ / 2) * fisher[layerIndex] * (Δweights) ** 2)
            return standard_loss + ewc_term
        return loss
    # Create the new model from that loss function
    new_model_ewc = Model.from_config(old_model.get_config())
    new_model_ewc.set_weights(deepcopy(old_model.get_weights()))
    new_model_ewc.compile(optimizer='adam', loss=ewc_loss(new_model_ewc), metrics=[dice_coef])
    # Train that model
    new_model_ewc.fit(X, Y, batch_size=batch_size, epochs=epochs)
    # Create a version of that model with a binary cross-entropy loss
    new_model_bce = Model.from_config(new_model_ewc.get_config())
    new_model_bce.set_weights(new_model_ewc.get_weights())
    new_model_bce.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    # Compute the new Fisher diagonal from that
    new_fisher = estimate_fisher_information(new_model_bce, X, Y, fisher_samples)
    # Save the model with binary cross-entropy loss (so no loading issues)
    new_model_bce.save(output_file)
    # Save the Fisher information matrix in the same file
    write_fisher(output_file, new_fisher)
    return new_model_bce

def add_fisher_to_existing_file(model_file, X, Y, fisher_samples=200):
    """
    Add the Fisher information matrix needed by train_model_incrementally to a
    network that has already been trained by a function other than train_model_from_scratch

    model_file:
        the path to an HDF5 file containing weights in Keras format, but no Fisher
        information matrix
    X:
        Inputs to the model for the task it has already been trained on (not a new task)
    Y:
        Outputs for the model for the task it has already been trained on (not a new task)
    fisher_samples: integer
        The number of samples to use for estimating the Fisher information matrix.
    """
    model = keras.models.load_model(model_file, custom_objects={'dice_coef': dice_coef})
    fisher = estimate_fisher_information(model, X, Y, num_samples=fisher_samples)
    write_fisher(output_file, fisher)
