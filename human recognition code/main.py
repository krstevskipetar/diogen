import os

import cv2
import numpy as np


if __name__ == '__main__':
    path = r'C:\Users\Koki\Downloads\IMG_9187.mp4'
    dst_path = r'C:\Users\Koki\Downloads\Sliki'

    capture = cv2.VideoCapture(path)
    _,frame = capture.read()

    cv2.VideoCapture(path)

    i=915
    while _:
        frame = frame[10:frame.shape[0]-10, 10:frame[0].shape[1]-10]
        # frame = cv2.resize(frame, (200,100), interpolation=cv2.INTER_NEAREST)
        frame = cv2.resize(frame, (625,375), interpolation=cv2.INTER_NEAREST)

        # cv2.imshow('frame',frame)
        # cv2.waitKey(0)
        print(frame.shape[0],frame.shape[1])

        frame_name = str(i).zfill(6) + '.png'
        if i%5==0:
            cv2.imwrite(os.path.join(dst_path,frame_name),frame)

        i+=1

        _, frame = capture.read()
