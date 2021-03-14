import os
import cv2
import requests
import tarfile
import numpy as np
import tensorflow as tf
    
from tqdm import tqdm
from core.label_map_util import get_label_map_dict

class ModelZoo:

    CenterNet_Resnet50_V1_FPN_512x512 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz'
    CenterNet_Resnet101_V1_FPN_512x512 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet101_v1_fpn_512x512_coco17_tpu-8.tar.gz'
    CenterNet_Resnet50_V2_512x512 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v2_512x512_coco17_tpu-8.tar.gz'
    CenterNet_MobileNetV2_FPN_512x512 = 'http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz'
    EfficientDet_D0_512x512	= 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
    EfficientDet_D1_640x640	= 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz'
    EfficientDet_D2_768x768	= 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz'
    EfficientDet_D3_896x896	= 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d3_coco17_tpu-32.tar.gz'
    EfficientDet_D4_1024x1024 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
    EfficientDet_D5_1280x1280 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz'	
    EfficientDet_D6_1280x1280 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz'
    EfficientDet_D7_1536x1536 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz'
    SSD_MobileNet_v2_320x320 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
    SSD_MobileNet_V1_FPN_640x640 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz'
    SSD_MobileNet_V2_FPNLite_320x320 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    SSD_ResNet50_V1_FPN_640x640_RetinaNet50 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'
    SSD_ResNet50_V1_FPN_1024x1024_RetinaNet50 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'	
    SSD_ResNet101_V1_FPN_640x640_RetinaNet101 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'
    SSD_ResNet101_V1_FPN_1024x1024_RetinaNet101 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'
    SSD_ResNet152_V1_FPN_640x640_RetinaNet152 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8.tar.gz'	
    SSD_ResNet152_V1_FPN_1024x1024_RetinaNet152 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz'	
    Faster_RCNN_ResNet50_V1_640x640 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet50_V1_1024x1024 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet50_V1_800x1333 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.tar.gz'
    Faster_RCNN_ResNet101_V1_640x640 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet101_V1_1024x1024 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet101_V1_800x1333 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_800x1333_coco17_gpu-8.tar.gz'
    Faster_RCNN_ResNet152_V1_640x640 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_640x640_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet152_V1_1024x1024 ='http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8.tar.gz'
    Faster_RCNN_ResNet152_V1_800x1333 = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet152_v1_800x1333_coco17_gpu-8.tar.gz'


    def __init__(self, model_name):
        self.model = self.__load_model(model_name)
        self.labels = self.__label_map()


    def load_image(self, image_path):
        img = cv2.imread(image_path)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_img)
        input_tensor = input_tensor[tf.newaxis, ...]
        return img, input_tensor


    def predict(self, input_tensor):
        output_dict = self.model.signatures["serving_default"](input_tensor)
        classes = output_dict["detection_classes"][0]
        scores = output_dict["detection_scores"][0]
        boxes = output_dict["detection_boxes"][0]
        return classes, scores, boxes


    def visualization(self, img, classes, scores, boxes, threshold=0.5):
        height, width = img.shape[:2]
        for idx, score in enumerate(scores):
            if score > threshold:
                class_id = int(classes[idx])
                box = boxes[idx]
                x1 = int(box[1] * width)
                y1 = int(box[0] * height)
                x2 = int(box[3] * width)
                y2 = int(box[2] * height)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, 1)
                cv2.putText(img, self.labels[class_id] + ":" + str(round(float(score), 2)), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 1)

        return img


    def __load_model(self, model_name):
        base_name = os.path.basename(model_name).split('.')[0]
        if os.path.isdir(os.path.join('model', base_name)):
            model_path = os.path.join('model', base_name, 'saved_model')
        else:
            model_name = self.__download_model(model_name)
            model_path = self.__unzip(model_name)
        model = tf.saved_model.load(model_path)
        return model


    def __download_model(self, url):
        block_size = 1024
        model_name = os.path.basename(url)
        target_path = os.path.join('model', model_name)
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            progress_bar.set_description("Download")
            with open(target_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
        return model_name


    def __unzip(self, file_path):
        target_path = os.path.join('model', file_path)
        with tarfile.open(target_path, "r:gz") as tar:
            progress_bar = tqdm(tar)
            progress_bar.set_description("Unzip Model")
            for tarinfo in progress_bar:
                tar.extract(tarinfo, 'model')
        os.remove(target_path)
        return os.path.join(target_path.split('.')[0], 'saved_model')


    def __label_map(self):
        label = dict()
        label_map = get_label_map_dict(os.path.join('core', 'mscoco_complete_label_map.pbtxt'), True)
        for item in label_map.items():
            label[item[1]] = item[0]
        return label