# coding:utf-8

import os


def transData(path, outfilename):
    S="23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    record={}
    for dir in os.listdir(path):
        if os.path.isdir(path+dir):
            l=os.listdir(path+dir)
            for i in l:
                record[int(i[:-4])]=dir
    f = open(outfilename,"w")
    for i,j in record.items():
        f.write(j+"\\"+str(i)+'.jpg '+str(S.index(j))+'\n')
    f.close()



#transData("./data/test/", "./data/test/test.txt")
#transData("./data/train/", "./data/train/train.txt")

transData("./data/testRGB/", "./data/testRGB/testRGB.txt")
transData("./data/trainRGB/", "./data/trainRGB/trainRGB.txt")