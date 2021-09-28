import cv2
import numpy

import threading ## Added for async
import pycuda.driver as cuda ## Added for async
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_with_plugins import TrtYOLO

NO_FRAME = numpy.zeros((360, 640), dtype=numpy.uint8)
NO_FRAME = cv2.putText(NO_FRAME, 'Sorry! No frame to show :(', (100, 200), 0, 1, (255, 255, 255), 3)
NO_FRAME = cv2.imencode('.jpg', NO_FRAME)[1].tobytes()

class TrtThread(threading.Thread):
    def __init__(self, condition, camera, args, threadQueue):
        threading.Thread.__init__(self)
        self.condition = condition
        self.camera = camera
        self.model = args.model
        self.category_num = args.category_num    ## Added For YOLO_async (09.13)
        self.letter_box = args.letter_box        ## Added For YOLO_async (09.13)
        self.threadQueue = threadQueue
        self.conf_th = 0.3
        self.cuda_ctx = None  # to be created when run
        self.trt_yolo = None   # to be created when run
        self.running = False

    def run(self):
        print('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(self.model, self.category_num, self.letter_box)         #Yolo문법: trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
        print('TrtThread: start running...')
        self.running = True

        while self.running:
            frame = self.camera.read()

            if frame is None:
                self.threadQueue.putThreadQueue(NO_FRAME, [])
            else:
                boxes, confs, clss = self.trt_yolo.detect(frame, self.conf_th)
                self.threadQueue.putThreadQueue(frame, boxes)

        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.camera.release()
        self.running = False
        self.join()


from queue import Queue

class ThreadQueue:

    def __init__(self):
        self.imgQueue = Queue(1)
        self.boxQueue = Queue(1)
    
    def putThreadQueue(self, img, boxes):
        self.imgQueue.put(img)
        self.boxQueue.put(boxes)
        return

    def getThreadQueue(self):
        return self.imgQueue.get(), self.boxQueue.get(), success

    def signalMainThread(self):
        self.imgQueue.task_done()
        return

    def isEmpty(self):
        return self.imgQueue.empty()        

    def destroy(self):
        del self.imgQueue
        return
