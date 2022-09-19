import argparse
import os
import time

import cv2

import scrfd
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='onnx/scrfd_2.5g_bnkps.onnx', help='model file path.')
    parser.add_argument('--source', default='img/1.jpg', help='image path, video path or webcam index')
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
                utils.draw_prediction(frame, bbox, conf, landmarks, args.line_thickness)

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
            utils.draw_prediction(img, bbox, conf, landmarks, args.line_thickness)
        else:
            print('No faces detected.')

        # Save image
        cv2.imwrite('result.jpg', img)
    else:
        raise ValueError('Wrong source.')
