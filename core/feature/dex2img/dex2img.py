import os
import time

from PIL import Image
import zipfile

import numpy as np

from config import logging

current_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('feature.dex2img')


def dex2img(apk_path, save_path, num_channels=3):
    """
    convert dex file to rbg images
    :param apk_path: an apk path
    :param save_path: a path for saving the resulting image
    :param num_channels: r, g, b channels
    :return: (status, save_path)
    """
    try:
        print("Processing " + apk_path)
        start_time = time.time()
        with zipfile.ZipFile(apk_path, 'r') as fh_apk:
            dex2num_list = []
            for name in fh_apk.namelist():
                if name.endswith('dex'):
                    with fh_apk.open(name, 'r') as fr:
                        hex_string = fr.read().hex()
                        dex2num = [int(hex_string[i:i + 2], base=16) for i in \
                                   range(0, len(hex_string), 2)]
                        dex2num_list.extend(dex2num)

        # extend to three channels (e.g., r,g,b)
        num_appending_zero = num_channels - len(dex2num_list) % num_channels
        dex2num_list += [0] * num_appending_zero
        # shape: [3, -1]
        dex2array = np.array([dex2num_list[0::3], dex2num_list[1::3], dex2num_list[2::3]], dtype=np.uint8)
        # get image matrix
        from math import sqrt, ceil
        _length = int(pow(ceil(sqrt(dex2array.shape[1])), 2))
        if _length > dex2array.shape[1]:
            padding_zero = np.zeros((3, _length - dex2array.shape[1]), dtype=np.uint8)
            dex2array = np.concatenate([dex2array, padding_zero], axis=1)
        dex2mat = np.reshape(dex2array, (-1, int(sqrt(_length)), int(sqrt(_length))))
        dex2mat_img = np.transpose(dex2mat, (1, 2, 0))
        img_handler = Image.fromarray(dex2mat_img)
        img_handler.save(save_path)
    except Exception as e:
        return e
    else:
        return save_path
