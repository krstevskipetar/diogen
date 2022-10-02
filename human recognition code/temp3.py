import os
import cv2


def data_rezie():
    pass


def img_resize(path):
    pass


path = r'C:/Users/Koki/Downloads/Sliki'
ann_path=r'C:/Users/Koki/Downloads/LSMT2022/Anotacii'
img_resize(path)
import numpy as np

images_list = []
ann_list = []
for img,ann in zip(os.listdir(path),os.listdir(ann_path)):

    if img[-4:] != '.png':
        continue
    print(img)
    img=os.path.join(path,img)
    im=cv2.imread(img)
    cv2.waitKey(0)
    cv2.imshow("im",im)

    im = cv2.resize(im, (625, 375), interpolation=cv2.INTER_AREA)
    cv2.imshow("Resized image", im)
    images_list.append(im)
    file = open(os.path.join(ann_path,ann), 'r')
    print
    file.readlines()
    for i in file.readlines():

