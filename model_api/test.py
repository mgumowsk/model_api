#from model_api.models import ClassificationModel
from build.nanobind.nanobind_model_api import ClassificationModel
#from build.pybind11.pybind11_model_api import ClassificationModel

import cv2
import time
import numpy as np

image = cv2.imread('/home/mgumowsk/dupa.jpg')

# Measure average time for model creation 10 times
total_creation_time = 0
for i in range(10):
    start_time = time.time()
    infer = ClassificationModel.create_model("/home/mgumowsk/openvino.xml")
    total_creation_time += (time.time() - start_time) * 1000

average_creation_time = total_creation_time / 10
print(f"Average model creation time over 10 runs: {average_creation_time} ms")


# Measure time for inference 10 times and calculate average time
total_time = 0
for i in range(100):
    start_time = time.time()
    result = infer(image)  # modified_image is numpy.ndarray
    total_time += (time.time() - start_time) * 1000

average_time = total_time / 100
print(f"Average inference time over 10 runs: {average_time} ms")

#print(f"Results: {result}")
