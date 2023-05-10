import os

import cv2

 
# Directory Config initialization 
inputhdatasetpath="COLOR DATASET/test"  
    
grayscaledataset="GRAYSCALE DATASET/test"

if not os.path.exists(grayscaledataset):
    os.makedirs(grayscaledataset)  
 
  
dir_list = os.listdir(inputhdatasetpath)
# Reads all the images from the directory and converts it to gray scale for training 
for dirname in dir_list:

    newdirpath=grayscaledataset+"//"+dirname 
    if not os.path.exists(newdirpath):
        os.makedirs(newdirpath)
    olddirpath=inputhdatasetpath+"//"+dirname 
    images_path = os.listdir(olddirpath) 
    imageno=1
    for n, image in enumerate(images_path):
        
        path=(os.path.join(olddirpath, image))
        
        image = cv2.imread(path)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filename=str(imageno)
        imagename=filename+".jpg"
        newimagepath=newdirpath+"//"+imagename 
        #print("newimagepath ",newimagepath)
        cv2.imwrite(newimagepath, grayscale)
        imageno=imageno+1
   
        
    print(olddirpath+" all images are converted in Grayscale \n\n")
    
 