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
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo, conf_th, vis, writer):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    
    #boxLength = 0
    runtime = 0.0
    fps = 0.0
    tic = time.time()

    while True:
        ret, frame = cap.read()
        if frame is None:  break
        boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        frame = vis.draw_bboxes(frame, boxes, confs, clss)
        writer.write(frame)
        print('.', end='', flush=True)

        boxLength = len(boxes)
        toc = time.time()
        runtime = toc - tic
        #print("run_algorithm  : ", '{:.2f}'.format(round((toc - tic) * 1000, 2)).rjust(7), "ms") ##
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

    print(" ")
    #print("box_length = " + str(boxLength))
    print("run  : ", '{:.2f}'.format(round(runtime * 1000, 2)).rjust(7), "ms")
    print("fps  : ", '{:.2f}'.format(round(fps, 2)).rjust(7), "fps")

    print('\nDone.')


def main():
    args = parse_args()
    
    conf_th = 0.3
    # number = 0

    # while conf_th < 10:
    print("conf_threshold = " + str(conf_th))

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

    loop_and_detect(cap, trt_yolo, conf_th=(conf_th), vis=vis, writer=writer)

    writer.release()
    cap.release()

    # conf_th += 1
    # number += 1
    # args.output = str(args.output[:-5] + str(conf_th) + ".mp4")



if __name__ == '__main__':
    main()
