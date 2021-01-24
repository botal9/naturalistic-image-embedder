from .conversions import *

import cv2
from enum import Enum
import numpy as np


class InsertionType(Enum):
    NAIVE = 0
    COLOR_TRANSFER = 1
    POISSON_BLENDING_NORMAL = 2
    POISSON_BLENDING_MIXED = 3


def insert_image(background_image_path,
                 foreground_image_path,
                 result_image_path,
                 offset,
                 insertion_type):
    bg = cv2.imread(background_image_path)
    fg = cv2.imread(foreground_image_path)
    fg_bitwise_mask = (fg > 0).astype('uint8')
    # fg_bitwise_mask = (255 * np.ones(fg.shape)).astype('uint8')

    if (insertion_type == InsertionType.POISSON_BLENDING_NORMAL or
            insertion_type == InsertionType.POISSON_BLENDING_MIXED):
        fg_h, fg_w = fg.shape[:2]
        offset_x, offset_y = offset
        center = (offset_x + fg_w // 2, offset_y + fg_h // 2)

        if insertion_type == InsertionType.POISSON_BLENDING_NORMAL:
            flags = cv2.NORMAL_CLONE
        elif insertion_type == InsertionType.POISSON_BLENDING_MIXED:
            flags = cv2.MIXED_CLONE
        else:
            flags = None
        result = cv2.seamlessClone(fg, bg, 255 * fg_bitwise_mask, center, flags)
    else:
        if insertion_type == InsertionType.COLOR_TRANSFER:
            bg_ref = crop(bg, fg.shape[:2], offset)
            fg_color_adjusted = mean_std_color_transfer(fg, fg_bitwise_mask, bg_ref)
        else:
            fg_color_adjusted = fg
        result = naive_insert(bg, fg_color_adjusted, fg_bitwise_mask, offset)
    cv2.imwrite(result_image_path, result)


def crop(image, size, offset):
    w, h = size
    offset_x, offset_y = offset
    return image[offset_y:offset_y+h, offset_x:offset_x+w, :]


def lighting_transfer(target, target_mask, reference):
    # Still works not as expected
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype('float32')
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype('float32')

    out = mean_std_color_transfer(target_lab, target_mask, reference_lab)

    out_bgr = cv2.cvtColor(out.astype('float32'), cv2.COLOR_LAB2BGR)
    return out_bgr.astype('uint8')


def mean_std_color_transfer(target, target_mask, reference):
    mask_size = np.sum(target_mask[:, :, 0])
    mean_target = np.sum(target, axis=(0, 1), keepdims=True) / mask_size
    mean_ref = np.mean(reference, axis=(0, 1), keepdims=True)

    target_h, target_w = target.shape[:2]
    target_mean_filled_frame = np.dstack([mean_target] * target_h * target_w).reshape(target_h, target_w, 3)
    framed_target = target + target_mean_filled_frame * (1 - target_mask)

    std_target = np.std(framed_target, axis=(0, 1), keepdims=True)
    std_ref = np.std(reference, axis=(0, 1), keepdims=True)
    out = (framed_target - mean_target) / std_target * std_ref + mean_ref
    out_clipped = np.clip(out, 0, 255)
    return out_clipped


def naive_insert(bg, fg, fg_mask, offset):
    fg_h, fg_w = fg.shape[:2]
    offset_x, offset_y = offset

    res = bg
    res[offset_y:offset_y+fg_h, offset_x:offset_x+fg_w, :] = \
        res[offset_y:offset_y+fg_h, offset_x:offset_x+fg_w, :] * (1 - fg_mask) + fg * fg_mask
    return res
