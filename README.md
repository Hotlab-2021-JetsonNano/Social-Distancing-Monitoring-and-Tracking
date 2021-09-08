# Social-Distancing-Monitoring-and-Tracking on AI Platform
This is a project which implement high-quality social-distancing monitoring system on NVIDIA Jetson Nano./
This allows tracking people on the video, categorizing person as 'definite risk' if he/she exceeds a 'threshold time' we set.

## Key features
### Model
YOLOv4
YOLOv4-tiny
YOLOv4-tiny-3L
(AP/FPS comparison datasheet)
Among these models, YOLOv4-tiny-3L showed the best balance between AP and FPS.
### TensorRT
The goal was implementing AI thermometer on embedded system using deep learning-based object detection model. So we optimized SSD(Single-shot multibox detector) via TensorRT, which is the library of NVIDIA for high-performance deep learning inference. Note that the TensorRT version which we used is 6.0.1
### Transfer Learning
Since the model only has to detect human, we did transfer learning of YOLOv4-tiny-3L with crowd-human dataset.
AP of the model improved.
(AP/FPS comparison datasheet)
### Human Tracking
<img src="demo/tracking.gif" width="640" height="324px"></img><br/>

## Environment
* Platform: Jetson Nano Developer Kit 4GB
* Camera : Logitech C270 webcam
* Libraries: TensorRT, OpenCV, NumPy, PyCUDA, etc. (version needed)

## Demo Monitoring


## Demo Tracking


## References
* TensorRT demo code : https://github.com/jkjung-avt/tensorrt_demos
* Crowd-Human Dataset : 
* Transfer-Learning demo code : 
* (Comparison) Social-Distancing using YOLOv5 : https://github.com/ChargedMonk/Social-Distancing-using-YOLOv5
* (Comparison) Social-Distancing Monitoring :  https://github.com/dongfang-steven-yang/social-distancing-monitoring
