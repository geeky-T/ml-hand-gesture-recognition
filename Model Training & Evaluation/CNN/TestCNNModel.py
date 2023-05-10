import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import mediapipe as mp

from model_layers import get_CNN_model

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

num_class = 5



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = get_CNN_model(num_class)
model.load_weights('model_CNN.h5')
    
    
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
    
# dictionary which assigns each label an emotion (alphabetical order)
# gesture_dict = {0: "BACK_LEFT",1: "BACK_RIGHT",2: "BLANK",3: "BUY_LEFT",4: "BUY_RIGHT",5: "MORE_LEFT",6: "MORE_RIGHT",7: "NEXT_LEFT",8: "NEXT_RIGHT",9: "PREVIOUS_LEFT",10: "PREVIOUS_RIGHT"}
gesture_dict = {0: "back",1: "buy", 2: 'more', 3: 'next', 4: 'previous'}
    
# start the webcam feed
cap = cv2.VideoCapture(1)
h=300
w=h

newfilepath="TEMP.jpg"

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    center_x = int(w/2)
    Gesturename = None
    center_y = int(h/2)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the x, y, and z coordinates of all the landmarks
            landmarks = hand_landmarks.landmark
            x = [landmark.x for landmark in landmarks]
            y = [landmark.y for landmark in landmarks]
            z = [landmark.z for landmark in landmarks]
            
            # Calculate the geometric center of the hand using the mean of the x, y, and z coordinates
            center_x = int((max(x) + min(x)) / 2 * frame.shape[1])
            center_y = int((max(y) + min(y)) / 2 * frame.shape[0])
    
        img = frame[int(center_y-(h/2)):int(center_y+(h/2)), int(center_x-(w/2)):int(center_x+(w/2))]
        cv2.rectangle(frame,(int(center_x-(w/2)),int(center_y-(h/2))),(int(center_x+(w/2)),int(center_y+(h/2))),(0,0,255),3)
    
        dim = (96, 96)
        try:
          img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        except:
          continue
        
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(img, (96, 96)), -1), 0)
        
        cv2.imwrite(newfilepath, img)
        
        
    
        prediction = model.predict(cropped_img)
        print(prediction)
        maxindex = int(np.argmax(prediction))
        Gesturename=gesture_dict[maxindex]
    else:
      Gesturename=gesture_dict[1]

    print("IG : ",Gesturename)
    cv2.putText(frame, Gesturename, (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          
    cv2.imshow('Capture Image( Press q to quit)',frame)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()