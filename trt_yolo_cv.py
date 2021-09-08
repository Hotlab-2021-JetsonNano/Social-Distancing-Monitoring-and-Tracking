"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.display import show_fps
from utils.distancing import show_distancing
from utils.distancing_class import FrameData


def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='output video file name')
    parser.add_argument(
        '-f', '--file', type=str, required=False,
        default='output_file/testsample.txt',
        help='output text file name')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=False,
        default='yolov4-tiny-crowd-416',
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo, conf_th, vis, writer, filePath):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    file = open(filePath, 'w')

    frameData = FrameData()
    frameData.set_timer()
    frameTime1 = 0.0
    frameTime2 = 0.0
    totalTime1 = time.time()
    isPrint = False

    while True:
        ret, frame = cap.read()
        if frame is None:  break

        if frameData.get_counter() == 329:
            isPrint = True

        modelTime1 = time.time()
        boxes, confs, clss = trt_yolo.detect(isPrint, frame, conf_th)
        modelTime2 = time.time()

        if isPrint:
            isPrint = False      

        frameTime1 = time.time()
        frame = show_distancing(frame, boxes, frameData)
        frameTime2 = time.time()

        frame = show_fps(frame, frameData.get_fps())
        
        writerTime1 = time.time()
        writer.write(frame)
        writerTime2 = time.time()
        file.write(frameData.get_log())

        frameData.increase_counter()
        frameData.update_fps()
        frameData.clear_log()

        print('.', end='', flush=True)

    file.close()

    totalTime2 = time.time()
    print("")
    #print("conf_thres : ", '{:.1f}'.format(conf_th))
    print("model : ", '{:.2f}'.format(round((modelTime2 - modelTime1) * 1000, 2)).rjust(10), "ms") ##
    print("algo  : ", '{:.2f}'.format(round((frameTime2 - frameTime1) * 1000, 2)).rjust(10), "ms") ##
    print("write : ", '{:.2f}'.format(round((writerTime2 - writerTime1) * 1000, 2)).rjust(10), "ms") ##
    print("")
    print("total : ", '{:.2f}'.format(round((totalTime2 - totalTime1), 2)).rjust(10), "s") ##
    print("fps   : ", '{:.2f}'.format(round(frameData.get_fps(), 2)).rjust(10), "fps") ##

    print('\nDone.')


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    filePath = args.file

    loop_and_detect(cap, trt_yolo, conf_th=0.3, vis=vis, writer=writer, filePath=filePath)

    writer.release()
    cap.release()


if __name__ == '__main__':
    main()
