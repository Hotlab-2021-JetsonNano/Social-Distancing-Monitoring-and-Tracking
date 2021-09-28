import os
import time
import argparse
import numpy
#import pdb

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_with_plugins import TrtYOLO
from utils.camera import add_camera_args, Camera
from utils.display import show_fps
from utils.distancing import show_distancing
from utils.distancing_class import FrameData

NO_FRAME = numpy.zeros((360, 640), dtype=numpy.uint8)
NO_FRAME = cv2.putText(NO_FRAME, 'Sorry! No frame to show :(', (100, 200), 0, 1, (255, 255, 255), 3)
NO_FRAME = cv2.imencode('.jpg', NO_FRAME)[1].tobytes()

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, default='yolov4-tiny-3l-crowd-416',
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
            'yolov4-csp|yolov4x-mish]-[{dimension}], where '
            '{dimension} could be either a single number (e.g. '
            '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


class YoloCamera:
    def __init__(self, VIDEO_SOURCE):
        args = parse_args()
        args.video = VIDEO_SOURCE

        self.trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

        self.camera = Camera(args)
        if not self.camera.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        self.frameData = FrameData()
        self.frameData.set_timer()

    def get_frame(self):
        frame = self.camera.read()

        if frame is None:
            self.camera.release()
            frame = NO_FRAME
        else:
            boxes, confs, clss = self.trt_yolo.detect(frame, 0.3)

            frame = show_distancing(frame, boxes, self.frameData)
            frame = show_fps(frame, self.frameData.get_fps())
            frame = cv2.imencode('.jpg', frame)[1].tobytes()

            self.frameData.increase_counter()
            self.frameData.update_fps()
            self.frameData.clear_log()

        return frame
