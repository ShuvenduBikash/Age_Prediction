import cv2 as cv
import glob
import numpy as np
import os


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

for mode in ['train','test']:
    for i in range(1, 100):
        path = mode+'\\'+str(i)+'\*'
        print("working on ", path)
        dest = 'imdb\\'+mode+'\\'+str(i)
        os.makedirs(dest, exist_ok=True)
        
        all_images = glob.glob(path)
        for i, image in enumerate(all_images):
            only_name = image.split('\\')[-1]
            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop = img[x:x + w, y:y + h]
                cv.imwrite(os.path.join(dest, only_name), crop)