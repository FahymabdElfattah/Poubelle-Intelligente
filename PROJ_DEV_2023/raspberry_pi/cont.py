from concetion import ObjectDetectionController

class ExternalClass:
    def __init__(self):
        self.obj_detection = ObjectDetectionController(
            model='efficientdet_lite0.tflite',
            camera_id=0,
            width=640,
            height=480,
            num_threads=4,
            enable_edgetpu=False
        )

    def run_object_detection(self):
        self.obj_detection.run()
        
        

    def print_category_name(self):
        category_name = self.obj_detection.get_category_name()
        if category_name:
            print("Category name:", category_name)
        else:
            print("No detection results.")


external_class = ExternalClass()
external_class.run_object_detection()
external_class.print_category_name()