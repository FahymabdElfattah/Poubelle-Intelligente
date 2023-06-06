from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
from Forgot import Forgot_password
from RPiFirestore import RPiFirestore
from Classification import Classification
import time
#from time import sleep



def login():
    
    username = username_entry.get()
    password = password_entry.get()
    user = RPiFirestore(username,password)
    clas = Classification()
    state = user.login()
    
    if state == True:
        if clas.openTrash() == True :
            
            time.sleep(20)
            clas.closTrash()
            classRate = clas.classifRate()
            time.sleep(10)
            user.sendClassif(classRate)
            
        messagebox.showinfo("Success", "Login successful!")
    else:
        messagebox.showerror("Error", "Invalid username or password.")

def forgot():
    #c1 = Forgot_password()
    #c1.forgot()
    messagebox.showinfo("Forgot password","Pour résoudre votre problème d'oubli de mot de passe, veuillez consulter notre agence")

window = Tk()

window.geometry('2166x7118')
window.resizable(0, 0)
window.state('normal')
window.title('Login Page')

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
