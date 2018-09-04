import cv2 as cv
import glob
import numpy as np
import os


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

all_images = glob.glob("CACD2000\\*")
dest_root = 'CACD_cropped\\'


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


for i in range(14, 63):
    createFolder('CACD_cropped\\train\\' + str(i))
    createFolder('CACD_cropped\\test\\' + str(i))

for i, image in enumerate(all_images):

    if i%1000==0:
        print(i)

    try:
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop = img[x:x + w, y:y + h]

        sub_dir = np.random.choice(['train\\', 'test\\'], p=[0.9, 0.1])
        dest_dir = dest_root + sub_dir + str(age) + "\\" + only_name

        cv.imwrite(dest_dir, crop)

    except:
        pass


# cv.imshow('img',crop)
# cv.waitKey(0)
# cv.destroyAllWindows()