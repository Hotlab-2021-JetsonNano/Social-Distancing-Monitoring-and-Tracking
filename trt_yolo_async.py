"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import pdb

import threading ## Added for async
from trt_yolo_async_class import ThreadQueue

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

import pycuda.driver as cuda ## Added for async

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.distancing import show_distancing
from utils.distancing_class import FrameData


WINDOW_NAME = 'TrtYOLODemo'
MAIN_THREAD_TIMEOUT = 20.0  # 20 seconds

## pipelining 을 위한 global 변수.
threadQueue = ThreadQueue()

class TrtThread(threading.Thread):
    """TrtThread

    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, condition, cam, model, category_num, letter_box, conf_th):
        """__init__

        # Arguments
            condition: the condition variable used to notify main
                       thread about new frame and detection result
            cam: the camera object for reading input image frames
            model: a string, specifying the TRT SSD model
            conf_th: confidence threshold for detection
        """
        threading.Thread.__init__(self)
        self.condition = condition
        self.cam = cam
        self.model = model
        self.conf_th = conf_th
        self.cuda_ctx = None  # to be created when run
        self.trt_yolo = None   # to be created when run
        self.running = False
        self.category_num = category_num    ## Added For YOLO_async (09.13)
        self.letter_box = letter_box        ## Added For YOLO_async (09.13)

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        print('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(self.model, self.category_num, self.letter_box)         #Yolo문법: trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
        print('TrtThread: start running...')
        self.running = True
        while self.running:
            tic1 = time.time()
            img = self.cam.read()
            if img is None:
                threadQueue.setImpossible()
                break

            trt_outputs, letter_box = self.trt_yolo.detect(img, self.conf_th)

            toc1 = time.time()
            print("child_thread : ", '{:.2f}'.format(round((toc1-tic1) * 1000, 2)).rjust(10), "ms")

            threadQueue.waitMainThread()
            threadQueue.putThreadQueue(img, trt_outputs, letter_box)
            
        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()


def loop_and_detect(condition ,cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    
    full_scrn = False
    # fps = 0.0
    # fps2 = 0.0
    # tic1 = time.time()
    frameData = FrameData()
    frameData.set_timer()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

        img, trt_outputs, letter_box = threadQueue.getThreadQueue()

        if threadQueue.isImpossible() and threadQueue.isEmpty():
            break

        threadQueue.signalMainThread()


        # toc1 = time.time()
        # print("child_thread : ", '{:.2f}'.format(round((toc1-tic1) * 1000, 2)).rjust(10), "ms")

        tic2 = time.time()
        boxes, confs, clss = trt_yolo.detect_post(img, conf_th, trt_outputs, letter_box)
        img = show_distancing(img, boxes, frameData)
        img = show_fps(img, frameData.get_fps())
        cv2.imshow(WINDOW_NAME, img)

        frameData.increase_counter()
        frameData.update_fps()
        frameData.clear_log()

        toc2 = time.time()
        print(" main_thread : ", '{:.2f}'.format(round((toc2-tic2) * 1000, 2)).rjust(10), "ms")

        # tic1 = time.time()
        

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, default=str('yolov4-tiny-3l-crowdhuman-416'),
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cuda.init()     ##Added for async (09.13), init pycuda driver

    cls_dict = get_cls_dict(args.category_num)
    #vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict) ## modified line(176->182)
    condition = threading.Condition() ## Added for async (09.14)

    trt_thread = TrtThread(condition, cam, args.model, args.category_num, args.letter_box, conf_th=0.3,)  ## Added for async (09.14) 
    trt_thread.start() #start the child thread
    loop_and_detect(condition, cam, trt_yolo, conf_th=0.3, vis=vis)
    trt_thread.stop() #stop the child thread


    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
