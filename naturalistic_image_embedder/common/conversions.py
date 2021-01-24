import numpy as np


def rgb_to_lms(image):
    conv_arr = np.array([[0.3811, 0.5783, 0.0402],
                         [0.1967, 0.7244, 0.0782],
                         [0.0241, 0.1288, 0.8444]])
    conv_arr_3d = np.tile(conv_arr, image.shape[:-1])
    res = conv_arr_3d * image.astype('float64')
    return res.astype('float64')


def lms_to_rgb(image):
    conv_arr = np.array([[4.4679, -3.5873, 0.1193],
                         [-1.2186, 2.3809, -0.1624],
                         [0.0497, -0.2439, 1.2045]])
    conv_arr_3d = np.tile(conv_arr, image.shape[:-1])
    res = conv_arr_3d * image.astype('float64')
    return res.astype('float64')


def lms_to_lab(image):
    image_log = np.log(image)
    conv_arr = (np.array([[3**0.5/3, 0.0, 0.0],
                          [0.0, 6**0.5/6, 0.0],
                          [0.0, 0.0, 2**0.5/2]]) *
                np.array([[1, 1, 1],
                          [1, 1, -2],
                          [1, -1, 0]]))
    conv_arr_3d = np.tile(conv_arr, image.shape[:-1])
    res = conv_arr_3d * image_log.astype('float64')
    return res.astype('float64')


def lab_to_lms(image):
    res = (np.array([[1, 1, 1],
                     [1, 1, -2],
                     [1, -1, 0]]) @
           np.array([[3**0.5/3, 0.0, 0.0],
                     [0.0, 6**0.5/6, 0.0],
                     [0.0, 0.0, 2**0.5/2]]) @
           image)
    res_exp = np.exp(res)
    return res_exp.astype('float64')



