# autonomous-quadcopter-control
Control a ArduCopter quadcopter with your hand using computer vision

## [Demo Video](https://drive.google.com/file/d/1Dgg7F-afjsjiCpskVet5e_PTHzKo0jkp/view?usp=sharing)

## Dependencies

- [`opencv-python`](https://pypi.org/project/opencv-python/)

- [`vJoy`](http://vjoystick.sourceforge.net/site/index.php/download-a-install/download)

- [`vjoy.py`](https://gist.github.com/Flandan/fdadd7046afee83822fcff003ab47087)

## Getting Started

1. Run the Python script on a separate PC with a camera

2. Connect to the drone and set throttle to the Y axis in joystick settings in Mission Planner

3. In the camera window, push `'s'` to start

4. Put your hand in the middle of the detection range, and push `'c'` to calibrate

5. Moving your hand above the calibration raises the virtual joystick's Y position, and vice versa. This is sent to Mission Planner to control the throttle of the drone.

6. Push `'q'` to quit

## Files

- `OpenCVDrone.py`: Main script, takes in camera input and outputs a virtual joystick position

- `control_althold` / `control_land`: Only edited to center the drone in an enclosed space, and to remove user pitch/roll/yaw control
