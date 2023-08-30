import argparse
import os
import platform
import time

import cv2
import numpy as np

import scrfd
import utils

# Global parameters
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='model file path.')
    parser.add_argument('--source', type=str, default='0', help='file/dir/webcam')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='cpu', help='[cpu, cuda, openvino, tensorrt]')
    parser.add_argument('--line-thickness', type=int, default=2, help='drawing thickness (pixels)')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    parser.add_argument('--draw-fps', action='store_true', help='Draw fps on the frame.')
    args = parser.parse_args()
    print(args)

    # Load detector
    detector = scrfd.SCRFD(args.model_path, args.conf_thres, args.iou_thres, args.device)

    # Inference
    # source: webcam or video
    if args.source.isnumeric() or args.source.lower().endswith(VID_FORMATS):
        if args.source.isnumeric():
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
        assert cap.isOpened()

        if args.source.isnumeric():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        count = 1
        accumulated_time = 0
        fps = 0
        while cv2.waitKey(5) != ord('q'):
            # Load image
            ret, img = cap.read()
            if not ret:
                break

            # Detect face
            start = time.perf_counter()
            pred = detector.detect_one(img)
            accumulated_time += (time.perf_counter() - start)
            if count % 10 == 0:
                fps = 1 / (accumulated_time / 10)
                accumulated_time = 0
            count += 1

            # Draw FPS
            if args.draw_fps:
                cv2.putText(img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'{fps:.2f} FPS', (10, 26), cv2.FONT_HERSHEY_DUPLEX,
                            0.66, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw prediction
            if pred is not None:
                bbox, conf, landmarks = detector.parse_prediction(pred)
                utils.draw_prediction(img, bbox, conf, landmarks, args.line_thickness, args.hide_conf)

            # Show prediction
            cv2.imshow('Face Detection', img)

        print('Quit inference.')
        cap.release()
        cv2.destroyAllWindows()

    # source: image
    elif args.source.lower().endswith(IMG_FORMATS):
        assert os.path.exists(args.source), f'Image not found: {args.source}'

        # Load image
        img: np.ndarray = cv2.imread(args.source)
        assert img is not None

        # Detect face
        pred = detector.detect_one(img)

        # Draw prediction
        if pred is not None:
            bbox, conf, landmarks = detector.parse_prediction(pred)
            utils.draw_prediction(img, bbox, conf, landmarks, args.line_thickness, args.hide_conf)
        else:
            print('No faces detected.')

        # Save image
        cv2.imwrite('result.jpg', img)
        print('Save result to "result.jpg"')
    else:
        raise ValueError(f'Wrong source: {args.source}')
