import cv2
import argparse

import pycuda.autoinit

from utils.camera import add_camera_args
from utils.yolo_with_plugins import TrtYOLO
from utils.distancing_class import FrameData
from utils.distancing import show_distancing

from flask import Flask, render_template, Response

app = Flask(__name__)

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

args = parse_args()
args.model = 'yolov4-tiny-3l-crowd-416'
args.video = 'source_video/people-640p.mp4'
args.category_num = 80

trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box, cuda_ctx=pycuda.autoinit.context)

camera = cv2.VideoCapture(args.video)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    frameData = FrameData()
    frameData.set_timer()

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            boxes, confs, clss = trt_yolo.detect(False, frame, 0.3)
            frame = show_distancing(frame, boxes, frameData)
            # frame = frameData.show_fps(frame)

            # ret, buffer = cv2.imencode('.jpg', frame)
            # frame = buffer.tobytes()
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        frameData.increase_counter()
        frameData.update_fps()
        frameData.clear_log()


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
