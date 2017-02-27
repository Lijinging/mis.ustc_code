import os
import shutil

def copy(path, label):
    shutil.copy(path, "data/test_images_rgb")

f = open("data/testRGB/testRGB.txt")
line = f.readline()
cnt = 0
while line and cnt<30:
    [path, label] = line.split( )
    copy("data/test/"+path.replace("\\","/"), label)
    #print path+","+label
    line = f.readline()
    cnt = cnt+1