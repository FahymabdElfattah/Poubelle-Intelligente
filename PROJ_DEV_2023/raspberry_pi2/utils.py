import cv2
import numpy as np
from tflite_support.task import processor

class PoseDetectionVisualizer:
    
    _MARGIN = 10  # pixels
    _ROW_SIZE = 10  # pixels
    _FONT_SIZE = 1
    _FONT_THICKNESS = 1
    _TEXT_COLOR = (0, 0, 255)  # red
    category_name = "non"
     

    def __init__(self):
        pass
    

    def visualize(self, image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.

        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualize.

        Returns:
            Image with bounding boxes.
        """
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, self._TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (
                self._MARGIN + bbox.origin_x, self._MARGIN + self._ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self._FONT_SIZE, self._TEXT_COLOR, self._FONT_THICKNESS)

        return image
    def get_category_name(self):
        return self.category_name
        
    
