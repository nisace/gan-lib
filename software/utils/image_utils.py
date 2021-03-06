import numpy as np
import scipy.misc


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    """ from https://github.com/carpedm20/DCGAN-tensorflow/blob/e4b395b3d31a4a89fb73dea405f6485a2795abb6/utils.py
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    x = x[j:j + crop_h, i:i + crop_w]
    return scipy.misc.imresize(x, [resize_h, resize_w])


def imread(path, is_grayscale=False):
    """ from https://github.com/carpedm20/DCGAN-tensorflow/blob/e4b395b3d31a4a89fb73dea405f6485a2795abb6/utils.py
    """
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def normalize(image):
    return np.array(image) / 127.5 - 1.


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, is_crop=True):
    """ from https://github.com/carpedm20/DCGAN-tensorflow/blob/e4b395b3d31a4a89fb73dea405f6485a2795abb6/utils.py
    """
    if is_crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image,
                                            [resize_height, resize_width])
    return normalize(cropped_image)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
    """ from https://github.com/carpedm20/DCGAN-tensorflow/blob/e4b395b3d31a4a89fb73dea405f6485a2795abb6/utils.py
    """
    image = imread(image_path, is_grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, is_crop)
