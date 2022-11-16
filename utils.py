import numpy as np
import tensorflow as tf
from PIL import Image


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1

    Parameters:
    ----------
    image : tensor
    J_style : scalar
        style cost

    Returns:
    -------
    tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image

    Parameters:
    ----------
    tensor : tensor

    Returns:
    -------
    Image: a PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
