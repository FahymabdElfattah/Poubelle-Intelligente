#install library on Rspberry pi
# pip3 install firebase_admin


# pinout

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from time import sleep

class RPiFirestore :
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    
    
    def __init__(self,myUesername,myPassword) :
        self.username = myUesername
        self.password = myPassword
#-----------------Login to user account -------------------------------
    def login(self):
        db = firestore.client()
        try :
            #___________________________ READ OPERATION ___________________________________
            readUser = db.collection('system').document(self.username).get()
            docDict = readUser.to_dict()
            userPassword = docDict['password']
            if(userPassword == self.password):
                return True
            else :
                return False
            #______________________________________________________________________________
        except Exception as e:
            print('Exception: ' + str(e))
            

#-----------------Send the classification rate to user account ---------
    def sendClassif(self,rate) :
        db = firestore.client()
        try :
            db.collection('system').document(self.username).update(
                {
                 'count' : rate
                 }
                )
            print('The sending of the price of classification success')
        except Exception as e:
            print('Exception: ' + str(e))
        
            
#___________________CREATE OPERATION________________________
#db.collection('outputDevices').document('digitalLED').set(   
#     {
#         'status' : False
#     }
# )

#db.collection('outputDevices').document('pwmLED').set(
#     {
#         'brightness' : 0
#     }
# )
#
#db.collection('outputDevices').document('buzzer').set(
#     {
#          'alarm' : False
#      }
#  )
# 
# db.collection('inputDevices').document('button').set(
#      {
#          'value' : False
#      }
# )
# #__________________________________________________________
# 
# 
# 
# while True:
#     #___________________________ READ OPERATION ___________________________________
#     readStatus = db.collection('outputDevices').document('digitalLED').get()
#     docDict = readStatus.to_dict()
#     LEDstatus = docDict['status']
# 
#     if(LEDstatus):
#         led1.on()
#         print(LEDstatus)
#     else :
#         led1.off()
#         print(LEDstatus)
#     #______________________________________________________________________________ 
# 
#     #_________________________UPDATE OPERATION____________________________________
#     if(btn.is_active):
#          db.collection('inputDevices').document('button').update(
#              {
#                  'value' : True
#              }
#          )
#     else:
#          db.collection('inputDevices').document('button').update(
#              {
#                  'value' : False
#              }
#          )
#     #___________________________________________________________________________
# 
# #__________________________DELETE OPERATION________________________________
# db.collection('outputDevices').document('buzzer').update(
#      {
#          'alarm' : firestore.DELETE_FIELD
#      }
# )
# #_________________________________________________________________________
# 
# 
