# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""

import argparse
import sys
import time

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red




glass = 0
metal=0
plastic = 0
carton = 0
person = 0
def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,) -> np.ndarray:
  
  global  glass 
  global  metal
  global plastic 
  global carton 
  global person 

  

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    if category_name == "glass":
        glass += 1
    if category_name == "metal":
        metal += 1
    if category_name == "plastic":
        plastic += 1
    if category_name == "carton":
        carton +=1 
    if category_name == "person":
        person += 1
    
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    
  return image


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Capture a single image from the camera and run object detection on it.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is an EdgeTPU model.
  """

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Capture a single image from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  
  success, image = cap.read()
  
  if not success:
    sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )

  # Convert the image from BGR to RGB as required by the TFLite model.
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Create a TensorImage object from the RGB image.
  input_tensor = vision.TensorImage.create_from_array(rgb_image)
  
  # Run object detection estimation using the model.
  detection_result = detector.detect(input_tensor)

  # Draw keypoints and edges on input image
  image = visualize(image, detection_result)
  rateG = 0
  rateM = 0
  rateP = 0
  rateC = 0
  ratePl = 0
  maximum = 0
  
  somme = glass + metal + carton + plastic + person
  rateP = (person * 100)/somme
  rateG = (glass * 100)/somme
  rateM = (metal * 100)/somme
  rateC = (carton * 100)/somme
  ratePl = (plastic * 100)/somme
  
  liste_rate = [rateP,rateG,rateM,rateC,ratePl]
  maximum = max(liste_rate)
  cout = (maximum*2)/100
  #print("La valeur maximale est :", maximum)

  if person != 0 :
      print("Le taux d'objets détectés de type personne est :",rateP,"%")
  if glass != 0 :
      print("Le taux d'objets détectés de type verre est :",rateG,"%")
  if plastic != 0 :
      print("Le taux d'objets détectés de type plastique est :",ratePl,"%")
  if metal != 0 :
      print("Le taux d'objets détectés de type métal est :",rateM,"%")
  if carton != 0 :
      print("Le taux d'objets détectés de type carton est :",rateC,"%")

  print("Votre solde après la classification de vos déchets est :", cout, "DH/1Kg")
  
  # Display the image with bounding boxes
  cv2.imshow('object_detector', image)
  cv2.waitKey(0)

  
  cv2.destroyAllWindows()
  


def main():
  
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))
  
#   somme = glass + metal + carton + plastic + person
#   rateP = (person * 100)/somme
#   print(rateP)


if __name__ == '__main__':
  
  main()
  
  
