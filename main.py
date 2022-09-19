import argparse
import os
import time

import cv2
import numpy as np

import scrfd


def draw_prediction(img: np.ndarray, bbox: list, conf: list, landmarks: list = None, thickness=2):
    """
    Draw prediction on the image.
    If the landmarks is None, only draw the bbox.
    """
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
        cv2.putText(img, f'{conf_one:.2f}', (x1, y1 - 2), 0, thickness / 3, conf_color, thickness, cv2.LINE_AA)

        # Draw landmarks
        if landmarks_one is not None:
            for point_x, point_y, color in zip(landmarks_one[::2], landmarks_one[1::2], landmarks_colors):
                cv2.circle(img, (point_x, point_y), thickness + 1, color, cv2.FILLED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='onnx/scrfd_2.5g_bnkps.onnx', help='model file path.')
    parser.add_argument('--source', default=0, help='image path, video path or webcam index')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--line-thickness', type=int, default=2, help='drawing thickness (pixels)')
    args = parser.parse_args()
    print(args)

    # Load detector
    detector = scrfd.SCRFD(args.model_file, args.conf_thres, args.iou_thres)

    # Inference
    if isinstance(args.source, int) or os.path.splitext(args.source)[1] == '.mp4':  # source: webcam or video
        cap = cv2.VideoCapture(args.source)
        assert cap.isOpened()

        if isinstance(args.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        count = 1
        accumulated_time = 0
        while cv2.waitKey(5) != ord('q'):
            # Load frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect face
            start = time.time()
            pred = detector.detect_one(frame)
            accumulated_time += (time.time() - start)
            if count % 10 == 0:
                print(f'FPS: {1 / (accumulated_time / 10):.2f}')
                accumulated_time = 0
            count += 1

            # Draw prediction
            if pred is not None:
                bbox, conf, landmarks = detector.parse_prediction(pred)
                draw_prediction(frame, bbox, conf, landmarks, args.line_thickness)

            # Show prediction
            cv2.imshow('Face detection', frame)

        print('Quit inference.')
        cap.release()
        cv2.destroyAllWindows()

    elif isinstance(args.source, str):  # source: image
        assert os.path.exists(args.source), f'Image not found: {args.source}'

        # Load image
        img = cv2.imread(args.source)
        assert img is not None

        # Detect face
        pred = detector.detect_one(img)

        # Draw prediction
        if pred is not None:
            bbox, conf, landmarks = detector.parse_prediction(pred)
            draw_prediction(img, bbox, conf, landmarks, args.line_thickness)
        else:
            print('No faces detected.')

        # Save image
        cv2.imwrite('result.jpg', img)
    else:
        raise ValueError('Wrong source.')
