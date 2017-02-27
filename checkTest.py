# coding:utf-8
import requests
import Image
import numpy as np
import os
import caffe
import shutil

#读取test.txt 获取path label Done
#对test中每一项读取 进行识别得到预测的值
#if label不等于预测值
#输出

labels = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

caffe.set_mode_gpu()
model_def = 'caffeconfig/deploy.prototxt'
model_weights = "caffeconfig/misustc_iter_1800.caffemodel"
net = caffe.Net(model_def, model_weights, caffe.TEST)  # 定义模型结构,包含了模型的训练权值,使用测试模式(不执行dropout)
net.blobs['data'].reshape(4,1,20,20)
net.reshape()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

def check(path, label):
    #预测path是什么 并与label比较
    im = caffe.io.load_image(path, False)
    transformed_image = transformer.preprocess('data', im)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output_prob = output['prob'][0]
    result = labels[output_prob.argmax()]
    if result!=labels[int(label)]:
        print result+", "+labels[int(label)]+" path:"+path

f = open("data/test/test.txt")
line = f.readline()
cnt = 0
while line:
    [path, label] = line.split( )
    check("data/test/"+path.replace("\\","/"), label)
    #print path+","+label
    cnt=cnt+1
    if cnt%400==0:
        print "count:",cnt
    line = f.readline()