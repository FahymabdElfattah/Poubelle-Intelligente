import argparse
import sys
import time

import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

#-------------------------------------
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from Forgot import Forgot_password
from RPiFirestore import RPiFirestore
import time
#from time import sleep
#------------------------------------------

#-----------------------OBJECT DETECTION --------------------

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + bbox.origin_x,
                     _MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

  return image


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  
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


# if __name__ == '__main__':
#   main()
# a = 2
# if a== 1 :
#     main()
# else :
#     print("vid")


#------------------------------------------------------------

window = Tk()

window.geometry('1166x4118')
window.resizable(0, 0)
window.state('normal')
window.title('Login Page')

#-----------------USER LOGIN -------------------------------
def login():
    username = username_entry.get()
    password = password_entry.get()
    user = RPiFirestore(username,password)
    state = user.login()
    
    if state == True:
        if clas.openTrash() == True :
            main()
            time.sleep(10)
            user.sendClassif(classRate)
            
        #messagebox.showinfo("Success", "Login successful!")
    else:
        messagebox.showerror("Error", "Invalid username or password.")

def forgot():
    c1 = Forgot_password()
    c1.forgot()
    messagebox.showinfo("????","Forgot password")
#------------------------------------------------------------
#------------------------------------------------------------


#-------------------------USER INTERFACE -------------------


# ============================background image============================
bg_frame = Image.open('background1.png')
photo = ImageTk.PhotoImage(bg_frame)

bg_panel = Label(window, image=photo)
bg_panel.image = photo
bg_panel.pack(fill='both', expand='yes')

# ====== Login Frame =========================
lgn_frame = Frame(window, bg='#040405', width=950, height=600)
lgn_frame.place(x=200, y=70)

# ====== WELCOME =======================================================
txt = "WELCOME"
heading = Label(lgn_frame, text=txt, font=('yu gothic ui', 25, "bold"), bg="#040405",
                             fg='white',
                             bd=5,
                             relief=FLAT)
heading.place(x=80, y=30, width=300, height=30)

# ============ Left Side Image ================================================
side_image = Image.open('vector.png')
photo = ImageTk.PhotoImage(side_image)
side_image_label = Label(lgn_frame, image=photo, bg='#040405')
side_image_label.image = photo
side_image_label.place(x=5, y=100)

# ============ Sign In Image =============================================
sign_in_image = Image.open('hyy.png')
photo = ImageTk.PhotoImage(sign_in_image)
sign_in_image_label = Label(lgn_frame, image=photo, bg='#040405')
sign_in_image_label.image = photo
sign_in_image_label.place(x=620, y=130)

# ============ Sign In label =============================================
sign_in_label = Label(lgn_frame, text="Sign In", bg="#040405", fg="white",
                                    font=("yu gothic ui", 17, "bold"))
sign_in_label.place(x=650, y=240)

# ============================username====================================
username_label = Label(lgn_frame, text="Username", bg="#040405", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
username_label.place(x=550, y=300)
username_entry = Entry(lgn_frame, highlightthickness=0, relief=FLAT, bg="#040405", fg="#6b6a69",
                                   font=("yu gothic ui ", 12, "bold"), insertbackground = '#6b6a69')
username_entry.place(x=580, y=335, width=270)
username_line = Canvas(lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
username_line.place(x=550, y=359)

# ===== Username icon ===================================================
username_icon = Image.open('username_icon.png')
photo = ImageTk.PhotoImage(username_icon)
username_icon_label = Label(lgn_frame, image=photo, bg='#040405')
username_icon_label.image = photo
username_icon_label.place(x=550, y=332)

# ============================login button================================
lgn_button = Image.open('btn1.png')
photo = ImageTk.PhotoImage(lgn_button)
lgn_button_label = Label(lgn_frame, image=photo, bg='#040405')
lgn_button_label.image = photo
lgn_button_label.place(x=550, y=450)
login = Button(lgn_button_label, text='LOGIN', font=("yu gothic ui", 13, "bold"), width=25, bd=0,
                            bg='#3047ff', cursor='hand2', activebackground='#3047ff', fg='white',command=login)
login.place(x=20, y=10)

# ============================Forgot password=============================
forgot_button = Button(lgn_frame, text="Forgot Password ?",
                                    font=("yu gothic ui", 13, "bold underline"), fg="white", relief=FLAT,
                                    activebackground="#040405"
                                    ,borderwidth=0, background="#040405", cursor="hand2",command=forgot)
forgot_button.place(x=630, y=510)

# ============================password====================================
password_label = Label(lgn_frame, text="Password", bg="#040405", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
password_label.place(x=550, y=380)
password_entry = Entry(lgn_frame, highlightthickness=0, relief=FLAT, bg="#040405", fg="#6b6a69",
                                    font=("yu gothic ui", 12, "bold"), show="*", insertbackground = '#6b6a69')
password_entry.place(x=580, y=416, width=244)
password_line = Canvas(lgn_frame, width=300, height=2.0, bg="#bdb9b1", highlightthickness=0)
password_line.place(x=550, y=440)

# ======== Password icon =================================================================================
password_icon = Image.open('password_icon.png')
photo = ImageTk.PhotoImage(password_icon)
password_icon_label = Label(lgn_frame, image=photo, bg='#040405')
password_icon_label.image = photo
password_icon_label.place(x=550, y=414)
window.mainloop()
