import os
import numpy as np
import cv2


if __name__ == '__main__':
    path = r'C:\Users\Koki\Downloads\LSMT2022\Anotacii'
    dest = r'C:\Users\Koki\Downloads\LSMT2022\Anotacii\krajni'

    xx=0
    for z in os.listdir(path):
        f = open(os.path.join(path,z), "r+").readlines()
        file = [i.strip('\n').split(',') for i in f]

        print(file)

        for i in file:
            count =0
            frame_name = "final"+str(xx).zfill(6) + '.txt'

            f = open(os.path.join(dest,frame_name), "a")
            for num in i:
                count +=1

                if count != 4:
                    print(num, count)
                    f.write(str(int(num)/1.684) + ",")
                else:
                    f.write(str(int(num) / 1.684))
            f.write('\n')
        f.close()
        xx+=1



