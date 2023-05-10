import os
import numpy as np
import cv2

# Directory Config initialization 
inputhdatasetpath="COLOR_DATASET/train"      
skindataset="SKIN_DATASET/train"
if not os.path.exists(skindataset):
    os.makedirs(skindataset)  
 
  
dir_list = os.listdir(inputhdatasetpath)
# Reads all the images from the directory and converts it to gray scale for training 
for dirname in dir_list:
    newdirpath=skindataset+"//"+dirname 
    if not os.path.exists(newdirpath):
        os.makedirs(newdirpath)

    olddirpath=inputhdatasetpath+"//"+dirname
    images_path = os.listdir(olddirpath) 
    imageno=1
    for n, image in enumerate(images_path):
        
        path=(os.path.join(olddirpath, image))
        
        image = cv2.imread(path)
        try:
          grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
          print(olddirpath, images_path, path)
          exit(0)
          
        filename=str(imageno)
        imagename=filename+".jpg"
        newimagepath=newdirpath+"//"+imagename 
        img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        #skin color range for hsv color space 
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
        YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
       
        YCrCb_result = cv2.bitwise_not(YCrCb_mask)
        cv2.imwrite(newimagepath, YCrCb_result)
        imageno=imageno+1
   
        
    print(olddirpath+" all images are converted in Skin Color \n\n")
    
 