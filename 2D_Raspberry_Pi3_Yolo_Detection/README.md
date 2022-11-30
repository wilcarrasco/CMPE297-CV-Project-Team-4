Requires Python 3.7, pytorch 1.7, torchvision 0.8

* Stable builds of pytorch & torchvision only support x86 CPU architectures
* Pi's ARM processor needs a custom build, provided as part of this repository:
* https://github.com/jordan-johnston271/yolov5-on-rpi4-2020.git

To run, execute through libcamerify, which supports legacy PiCamera2 with later kernel versions:
`libcamerify python pi3_yolov5_gui.py`
