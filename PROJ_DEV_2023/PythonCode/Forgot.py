from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox

class Forgot_password :
    def forgot(self):
        root = Tk()
        root.configure(bg="black") # Set the background color to black
        lgn_frame = Frame(root, bg='#040405', width=950, height=600)
        lgn_frame.place(x=200, y=70)

        # Set the size and position of the window
        #root.geometry("400x300+200+200")
        username_label = Label(lgn_frame, text="Username", bg="#040405", fg="#4f4e4d",
                                    font=("yu gothic ui", 13, "bold"))
        username_label.place(x=550, y=300)

        # Add widgets or components here
        # ...

        root.mainloop()

        
        
