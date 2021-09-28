import argparse
from utils.camera import add_camera_args

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
        '-m', '--model', type=str,
        default='yolov4-tiny-3l-crowd-416',
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


###
from queue import Queue

class ThreadQueue:

    def __init__(self):
        self.possible = True
        self.imgQueue = Queue(1)
        self.boxQueue = Queue(1)

    def waitMainThread(self):
        #self.check.join()
        return
    
    def putThreadQueue(self, img, boxes):
        self.imgQueue.put(img)
        self.boxQueue.put(boxes)
        return

    def getThreadQueue(self):
        success = not self.isImpossible() or not self.isEmpty()
        if not success:
            return None, None, success
        return self.imgQueue.get(), self.boxQueue.get(), success

    def signalMainThread(self):
        self.imgQueue.task_done()
        return

    def setImpossible(self):
        self.possible = False
        return

    def isImpossible(self):
        return self.possible != True

    def isPossible(self):
        return self.possible

    def isEmpty(self):
        return self.imgQueue.empty()        

    def destroy(self):
        del self.imgQueue
        return
  
