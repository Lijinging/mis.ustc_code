# coding:utf-8
import requests
import Image
import numpy as np
import os
import caffe
import shutil

if not os.path.exists("image"):
    os.mkdir("image")

table = [i if i < 140 else 255 for i in xrange(256)]
labels = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

for i in labels:
    if not os.path.exists("image/"+i):
        os.mkdir("image/"+i)

caffe.set_mode_gpu()
model_def = 'caffeconfig/deploy.prototxt'
model_weights = "caffeconfig/misustc_iter_1900.caffemodel"
net = caffe.Net(model_def, model_weights, caffe.TEST)  # 定义模型结构,包含了模型的训练权值,使用测试模式(不执行dropout)
net.blobs['data'].reshape(4,1,20,20)
net.reshape()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

def getImage():
    url = "http://mis.teach.ustc.edu.cn/randomImage.do?data=\'" + str(np.random.randint(1000000, 9999999)) + "\'"
    req = requests.get(url)
    try:
        with open("image/tmp.jpg", 'wb') as file:
            file.write(req.content)
            file.close()
    except IOError:
        print "IOError"
    req.close()
    img = Image.open("image/tmp.jpg").convert('L').point(table)
    img.crop((00, 0, 20, 20)).save("image/tmp0.jpg")
    img.crop((20, 0, 40, 20)).save("image/tmp1.jpg")
    img.crop((40, 0, 60, 20)).save("image/tmp2.jpg")
    img.crop((60, 0, 80, 20)).save("image/tmp3.jpg")

def classify():
    global index
    for i in range(4):
        filename = "image/tmp"+str(i)+".jpg"
        im = caffe.io.load_image(filename, False)
        transformed_image = transformer.preprocess('data', im)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]
        result = labels[output_prob.argmax()]
        shutil.move(filename, "image/"+result+"/"+str(index)+".jpg")
        index+=1
index = 4000
while index<16000:
    getImage()
    classify()
    if index%400==0:
        print "------index:"+str(index)+"------"
os.remove("image/tmp.jpg")



