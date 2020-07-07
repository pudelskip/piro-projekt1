import argparse
import cv2
import os
import numpy as np
from utils import process
from compare import compareSet


def loadImage(path,k):
    fileName = str(k)+'.png'
    imagePath = os.path.join(path,fileName)
    img = cv2.imread(imagePath)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIRO')
    parser.add_argument("path")
    parser.add_argument("N", type=int)
    args = parser.parse_args()
    path = args.path.replace('\'', '')
    path = path.replace('\"','')
    setDict = {}
    truth = []
    corr = 'correct.txt'
    for i in range(args.N):
        a = loadImage(path,i)
        try:
            v = process(a,i)
        except:
            v= np.zeros(57)
        setDict[i]=v
    compareSet(setDict,truth)
