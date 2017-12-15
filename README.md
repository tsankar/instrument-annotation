# Instrument Annotation Scratch Space

This repo houses miscellaneous code from my project with Prof. Serge Belongie at Cornell on musical instrument detection and annotation using computer vision and deep learning. There will be a separate repository for the official project code.

```module/``` holds an interface for using SSD detection on videos using the fine tuned model for instrument detection. ```module/demo.py``` provides a demo of this interface. The weights are not hosted in this repo as the file is too large.

Note that the module has a dependency on SSD caffe, which can either be installed from the official repo [here](https://github.com/weiliu89/caffe/tree/ssd) or from my fork which I used for this work [here](https://github.com/tsankar/SSD-instruments). Follow the instructions in the README to install caffe from one of those repositories.
