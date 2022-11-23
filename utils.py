from typing import Tuple

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


def clip_coords(boxes: np.ndarray, shape: tuple) -> np.ndarray:
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    boxes[:, 5:15:2] = boxes[:, 5:15:2].clip(0, shape[1])  # x axis
    boxes[:, 6:15:2] = boxes[:, 6:15:2].clip(0, shape[0])  # y axis
    return boxes


def draw_prediction(img: np.ndarray, bbox: list, conf: list, landmarks: list = None, thickness=2, hide_conf=False):
    # Draw prediction on the image. If the landmarks is None, only draw the bbox.
    assert img.ndim == 3, f'img dimension is invalid: {img.ndim}'
    assert img.dtype == np.uint8, f'img dtype must be uint8, got {img.dtype}'
    assert img.shape[-1] == 3, 'Pass BGR images. Other Image formats are not supported.'
    assert len(bbox) == len(conf), 'bbox and conf must be equal length.'
    if landmarks is None:
        landmarks = [None] * len(bbox)
    assert len(bbox) == len(conf) == len(landmarks), 'bbox, conf, and landmarks must be equal length.'

    bbox_color = (0, 255, 0)
    conf_color = (0, 255, 0)
    landmarks_colors = ((0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))
    for bbox_one, conf_one, landmarks_one in zip(bbox, conf, landmarks):
        # Draw bbox
        x1, y1, x2, y2 = bbox_one
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness, cv2.LINE_AA)

        # Text confidence
        if not hide_conf:
            cv2.putText(img, f'{conf_one:.2f}', (x1, y1 - 2), None, 0.6, conf_color, thickness, cv2.LINE_AA)

        # Draw landmarks
        if landmarks_one is not None:
            for point_x, point_y, color in zip(landmarks_one[::2], landmarks_one[1::2], landmarks_colors):
                cv2.circle(img, (point_x, point_y), 2, color, cv2.FILLED)


def align_face(img: np.ndarray, landmarks: list) -> np.ndarray:
    landmarks = landmarks[0]  # Only use the first face
    landmarks = np.array([landmarks[0:2],  # left eye
                          landmarks[2:4],  # right eye
                          landmarks[4:6],  # nose
                          landmarks[6:8],  # left mouth
                          landmarks[8:10]])  # right mouse
    # equivalent to landmarks = np.reshape(landmarks, (-1, 2))

    # it is based on (112, 112) size img.
    dst = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], np.float32)

    # scikit-image
    # import skimage.transform
    # tform = skimage.transform.SimilarityTransform()
    # assert tform.estimate(landmarks, dst)
    # M = tform.params[:2]
    # warped_img = cv2.warpAffine(img, M, (112, 112))

    # OpenCV
    M, inliers = cv2.estimateAffinePartial2D(landmarks, dst, ransacReprojThreshold=np.inf)
    assert np.count_nonzero(inliers) == inliers.size
    warped_img = cv2.warpAffine(img, M, (112, 112))
    return warped_img
