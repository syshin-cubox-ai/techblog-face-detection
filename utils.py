from typing import Union, Tuple

import cv2
import numpy as np


def resize_preserving_aspect_ratio(img: np.ndarray, img_size: int, scale_ratio=1.0) -> Tuple[np.ndarray, float]:
    # Resize preserving aspect ratio. scale_ratio is the scaling ratio of the img_size.
    h, w = img.shape[:2]
    scale = img_size // scale_ratio / max(h, w)
    if scale != 1:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)
    return img, scale


def clip_coords(boxes: Union[np.ndarray], shape: tuple) -> Union[np.ndarray]:
    # MODIFIED for face detection
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    boxes[:, 5:15:2] = boxes[:, 5:15:2].clip(0, shape[1])  # x axis
    boxes[:, 6:15:2] = boxes[:, 6:15:2].clip(0, shape[0])  # y axis
    return boxes
