import argparse
import sys
import time

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


class ObjectDetectionController:
    def __init__(
        self,
        model: str = 'efficientdet_lite0.tflite',
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        num_threads: int = 4,
        enable_edgetpu: bool = False
    ):
        self.model = model
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.num_threads = num_threads
        self.enable_edgetpu = enable_edgetpu

        # Initialize the object detection model
        base_options = core.BaseOptions(
            file_name=self.model,
            use_coral=self.enable_edgetpu,
            num_threads=self.num_threads
        )
        detection_options = processor.DetectionOptions(
            max_results=3,
            score_threshold=0.3
        )
        self.options = vision.ObjectDetectorOptions(
            base_options=base_options,
            detection_options=detection_options
        )
        self.detector = None

    def initialize(self):
        """Initialize the object detection model."""
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def visualize(
        self,
        image: np.ndarray,
        detection_result: processor.DetectionResult,
    ) -> np.ndarray:
        """Draw bounding boxes on the input image and return it."""
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            print("Category name:", category_name)
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (10 + bbox.origin_x, 10 + 10 + bbox.origin_y)
            cv2.putText(
                image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                1
            )

        return image

    def run(self) -> None:
        """Capture a single image from the camera and run object detection on it."""
        # Initialize the object detection model
        self.initialize()

        # Capture a single image from the camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = self.detector.detect(input_tensor)

        # Draw keypoints and edges on input image
        image = self.visualize(image, detection_result)

        # Display the image with bounding boxes
        cv2.imshow('object_detector', image)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    def get_category_name(self):
        detection_result = self.run()
        if detection_result and detection_result.detections:
            category = detection_result.detections[0].categories[0]
            return category.category_name
        else:
            return None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', required=False, default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
    args = parser.parse_args()

    obj_detection = ObjectDetectionController(
        model=args.model,
        camera_id=int(args.cameraId),
        width=args.frameWidth,
        height=args.frameHeight,
        num_threads=int(args.numThreads),
        enable_edgetpu=bool(args.enableEdgeTPU)
    )
    obj_detection.run()


if __name__ == '__main__':
    main()
