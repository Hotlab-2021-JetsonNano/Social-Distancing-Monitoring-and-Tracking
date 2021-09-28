from server_async_core import YoloCamera as camera
from flask import Flask, render_template, Response

VIDEO_SOURCE = 'source_video/people-640p.mp4'

app = Flask(__name__)


def gen_frames(camera):  # generate frame by frame from camera
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(camera(VIDEO_SOURCE)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
