import argparse

import cv2
import numpy
import pycuda.autoinit  # This is needed for initializing CUDA driver

import threading ## Added for async
from server_async_core_thread import TrtThread, ThreadQueue

from utils.camera import add_camera_args, Camera
from utils.display import show_fps
from utils.distancing import show_distancing
from utils.distancing_class import FrameData

def parse_args(VIDEO_SOURCE):
        parser = argparse.ArgumentParser()
        parser = add_camera_args(parser)
        parser.add_argument(
            '-c', '--category_num', type=int, required=False, default=80,
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

        if VIDEO_SOURCE == 0:
            args.usb = VIDEO_SOURCE
        else:
            args.video = VIDEO_SOURCE
            
        return args
        

class YoloCamera:
    def __init__(self, VIDEO_SOURCE):
        args = parse_args(VIDEO_SOURCE)

        camera = Camera(args)
        if not camera.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        condition = threading.Condition() ## Added for async (09.14)
        self.threadQueue = ThreadQueue()
        self.trt_thread = TrtThread(condition, camera, args, self.threadQueue)  ## Added for async (09.14) 

        self.frameData = FrameData()
        self.frameData.set_timer()

        BLACK_FRAME = numpy.zeros((360, 640), dtype=numpy.uint8)
        BLACK_FRAME = cv2.putText(BLACK_FRAME, 'Sorry! No frame to show :(', (100, 200), 0, 1, (255, 255, 255), 3)
        self.NO_FRAME = cv2.imencode('.jpg', BLACK_FRAME)[1].tobytes()

    def thread_start(self):
        self.trt_thread.start()
        return

    def thread_stop(self):
        self.trt_thread.stop()
        return

    def get_frame(self):
        frame, boxes, success = self.threadQueue.getThreadQueue()
        self.threadQueue.signalMainThread()

        if not success:
            return self.NO_FRAME

        frame = show_distancing(frame, boxes, self.frameData)
        frame = show_fps(frame, self.frameData.get_fps())
        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        self.frameData.increase_counter()
        self.frameData.update_fps()
        self.frameData.clear_log()

        return frame
