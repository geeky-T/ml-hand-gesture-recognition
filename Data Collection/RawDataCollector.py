import cv2
import os
import numpy as np
import mediapipe as mp
import random

# creates MP Hand object that helps identify hand in a video feed
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Folder initialization to store images
type_datasetpaths = ['train', 'test']
datasetpath="COLOR_DATASET"
gesture_type="next"

if not os.path.exists(datasetpath):
    os.makedirs(datasetpath)         
    


for typename in type_datasetpaths:
  type_datasetpath=datasetpath+"//"+typename
  if not os.path.exists(type_datasetpath):
      os.makedirs(type_datasetpath)      
  dataset_gesture_type=type_datasetpath+"//"+gesture_type
  if not os.path.exists(dataset_gesture_type):
    os.makedirs(dataset_gesture_type)          

    
# Image Collection Config initialization    
cap=cv2.VideoCapture(1)
h=300
w=300
collect=False
testCount = 0
trainCount = 0
captured_type = ''
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MP Hand object identified hand in the video feed
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    center_x = int(w/2)
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

        
    # Crops the image from the video feed based on the required size around the hand
    img = frame[int(center_y-(h/2)):int(center_y+(h/2)), int(center_x-(w/2)):int(center_x+(w/2))]
      
    # Draws a rectangle around the hand and pushes it to live feed for user to see what part will be collected.
    cv2.rectangle(frame,(int(center_x-(w/2)),int(center_y-(h/2))),(int(center_x+(w/2)),int(center_y+(h/2))),(0,0,255),3)
    
    cv2.imshow('Capture Image( Press ESC to quit)',frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
      collect = not collect
    if collect:
    # Saving of image (Randomly assigns the image as test or train type based on ratio of train to test of 4: 1).
        filename = ''
        random_number = random.randint(1, 10)
        dataset_gesture_type = None
        if(random_number > 8):
          dataset_gesture_type = datasetpath+"//test//"+gesture_type
          captured_type='test'
          testCount+=1
          filename = str(testCount)
        else:
          dataset_gesture_type = datasetpath+"//train//"+gesture_type
          captured_type='train'
          trainCount+=1
          filename = str(trainCount)
        dim = (96, 96)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        newfilepath=dataset_gesture_type+"//"+filename+"_right.jpg"
        cv2.imwrite(newfilepath,img)
        img = np.fliplr(img)
        newfilepath=dataset_gesture_type+"//"+filename+"_left.jpg"
        cv2.imwrite(newfilepath,img)
        print(f"Captured path is {captured_type}:", filename)
      
 
cap.release()
cv2.destroyAllWindows()