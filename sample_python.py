# Using GPU computing
import tensorflow as tf
physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], enable=True)

# Model Prediction
from core.detection import ModelZoo
model = ModelZoo(ModelZoo.SSD_MobileNet_v2_320x320)
img, input_tensor = model.load_image('./images/dog.jpg')
classes, scores, boxes = model.predict(input_tensor)
visual = model.visualization(img, classes, scores, boxes, 0.7)

# OpenCV Visualization
import cv2
cv2.imshow("visual", visual)
cv2.waitKey()