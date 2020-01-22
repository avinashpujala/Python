from skimage.util import invert
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\woottenm\FFMPEG\bin\ffmpeg.exe'

import sys
sys.path.append(r'C:\Users\woottenm\Documents\code\zebrafish-analysis')
from image_process import brightest_portion
from snippets.assorted import map_np

def animate_image_sequence(images, interval=20, show_frame=False):
    return animate_image_sequences(
        images,
        interval=interval,
        show_frame=show_frame
    )

def animate_image_sequence_pair(images_1, images_2, interval=20, show_frame=False):
    return animate_image_sequences(
        images_1,
        images_2,
        interval=interval,
        show_frame=show_frame
    )

def animate_image_sequences(*images, **kwargs):
    """
    Animate image sequences using ``matplotlib`` in Jupyter notebooks.

    Parameters
    ----------
    images : any number of ndarray (time * rows * columns)
        Sequences of images to use in the animations
    interval : int (optional, default = 20)
        The time to wait between frames, in milliseconds
    show_frame : bool (optional, default = True)
        Whether to show the frame counter
    cmap : str (optional, default = 'viridis')
        Colormap for Matplotlib
    """
    interval = kwargs.get('interval', 20)
    show_frame = kwargs.get('show_frame', True)
    cmap = kwargs.get('cmap', 'viridis')

    fig, axes = plt.subplots(1, len(images), squeeze=False)
    axes = axes[0]
    plt_images = [None] * len(images)

    def init():
        for plot_index in range(len(images)):
            axes[plot_index].axis('off')
            plt_images[plot_index] = axes[plot_index].imshow(images[plot_index][0], cmap=cmap)
        for plot_index in range(len(images)):
            plt_images[plot_index].set_data(images[plot_index][0])
        return tuple(plt_images)

    def animate(i):
        for plot_index in range(len(images)):
            plt_images[plot_index].set_data(images[plot_index][i])
        if show_frame:
            plt.title('Frame {:04d}'.format(i))
        return tuple(plt_images)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames = len(images[0]),
        interval=interval,
        blit=True
    )
    return HTML(anim.to_html5_video())

def fade_into_existence(image, frame_count=1000, show_frame=True):
    eps = np.finfo(float).resolution
    inv = invert(image)
    fraction = np.linspace(eps, 1, frame_count)
    cutoff_indices = np.asarray((1 - fraction) * image.size, dtype=int)
    cutoffs = np.sort(inv.ravel())[cutoff_indices]
    frames = map_np(lambda cutoff: inv >= cutoff, cutoffs)
    print(frames.shape)
    return animate_image_sequence(frames, show_frame=show_frame)
