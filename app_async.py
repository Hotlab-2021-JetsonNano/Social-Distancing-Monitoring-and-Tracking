import cv2

import pycuda.autoinit
import pycuda.driver as cuda ## Added for async
import threading ## Added for async

from utils.yolo_with_plugins import TrtYOLO
from utils.distancing_class import FrameData
from utils.distancing import show_distancing
from app_async_class import parse_args, ThreadQueue
from utils.camera import Camera

from flask import Flask, render_template, Response


args = parse_args()
args.model = 'yolov4-tiny-3l-crowd-416'
args.video = 'source_video/people-640p.mp4' # 0 if webcam
args.category_num = 80

MAIN_THREAD_TIMEOUT = 20.0  # 20 seconds
threadQueue = ThreadQueue()

app = Flask(__name__)

class TrtThread(threading.Thread):

    def __init__(self, condition, cam, model, category_num, letter_box, conf_th):
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
        print('TrtThread: loading the TRT YOLO engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_yolo = TrtYOLO(self.model, self.category_num, self.letter_box, self.cuda_ctx)         #Yolo문법: trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
        print('TrtThread: start running...')
        self.running = True
        
        while self.running:
            img = self.cam.read()

            if img is None:
                threadQueue.setImpossible()
                break

            boxes, confs, clss = self.trt_yolo.detect(img, self.conf_th)

            threadQueue.putThreadQueue(img, boxes)
            
        del self.trt_yolo
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()


def gen_frames():  # generate frame by frame from camera
    frameData = FrameData()
    frameData.set_timer()

    while True:
        frame, boxes = threadQueue.getThreadQueue()
        threadQueue.signalMainThread()

        frame = show_distancing(frame, boxes, frameData)
        # frame = frameData.show_fps(frame)

        frameData.increase_counter()
        frameData.update_fps()
        frameData.clear_log()

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    condition = threading.Condition() ## Added for async (09.14)
    camera = Camera(args)  # use 0 for web camera
    trt_thread = TrtThread(condition, camera, args.model, args.category_num, args.letter_box, conf_th=0.3)  ## Added for async (09.14) 
    trt_thread.start() #start the child thread

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
    trt_thread.stop()
