# coding:utf-8
from PIL import Image
import os

labels = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def conv(fromDir, toDir):
    os.mkdir(toDir)
    for dir in labels:
        os.mkdir(toDir + dir)
        for f in os.listdir(fromDir + dir):
            Image.open(fromDir + dir + "/" + f).convert("RGB").save(toDir + dir + "/" + f)


conv("data/test/", "data/testRGB/")
conv("data/train/", "data/trainRGB/")
