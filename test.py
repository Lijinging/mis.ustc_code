# coding:utf-8
import caffe
import Image
import requests
import os
import numpy as np
import matplotlib.pyplot as plt

#获取验证码

url = "http://mis.teach.ustc.edu.cn/randomImage.do?data=\'"+str(np.random.randint(1000000,9999999))+"\'"
req = requests.get(url)

#处理验证码
table = [i if i <140 else 255 for i in xrange(256)]
if not os.path.exists("image"):
    os.mkdir("image")
try:
    with open("image/tmp.jpg", 'wb') as imagefile:
        imagefile.write(req.content)
        imagefile.close()
except IOError:
    print "IOError"
    exit(1)

req.close()
img = Image.open("image/tmp.jpg").convert('L').point(table)
#img.show()
img.crop((0, 0, 20, 20)).save("image/part0.jpg")
img.crop((20, 0, 40, 20)).save("image/part1.jpg")
img.crop((40, 0, 60, 20)).save("image/part2.jpg")
img.crop((60, 0, 80, 20)).save("image/part3.jpg")

#配置caffe
caffe.set_mode_gpu()
model_def = 'caffeconfig/deploy.prototxt'
model_weights = "caffeconfig/misustc_iter_1800.caffemodel"
net = caffe.Net(model_def, model_weights, caffe.TEST)  # 定义模型结构,包含了模型的训练权值,使用测试模式(不执行dropout)
net.blobs['data'].reshape(4,1,20,20)
net.reshape()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

#准备识别
result = []
labels = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
needShow = True

#输入数据
for i in range(4):
    im = caffe.io.load_image("image/part"+str(i)+".jpg", False)
    transformer_image = transformer.preprocess('data', im)
    net.blobs['data'].data[i, ...] = transformer_image

#进行预测
output = net.forward()

#显示输出结果
def vis_square(data):   #显示卷积操作的结果

    # 正则化数据
    data = (data - data.min()) / (data.max()-data.min())
    # 将滤波器的核转变为正方形
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
    # (0, 1), (0, 1)) 在相邻的滤波器之间加入空白
    # + ((0, 0),) * (data.ndim - 3) # 不扩展最后一维
    data = np.pad(data, padding, mode='constant', constant_values=1)  # 扩展一个像素(白色)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if data.ndim == 3:
        data = data[:, :, 0]
    plt.imshow(data, cmap="gray")
    plt.axis('off')
    plt.show()

def showStatus():
    for i in range(4):
        # 前四行中最大值为概率最大，认为最终结果
        output_prob = output['prob'][i]
        # 将最终结果写入result中
        result.append(labels[output_prob.argmax()])
        # 输出识别结果
        print 'predicted class and label:', output_prob.argmax(), labels[output_prob.argmax()]
        # 查看其他几个置信较高的结果
        top_inds = output_prob.argsort()[::-1][:-5]
        print 'probabilites and labels:',zip(output_prob[top_inds], np.array(list(labels))[top_inds])
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + "\t" + str(blob.data.shape)
    print
    for layer_name, param in net.params.iteritems():
        print layer_name + "\t" + str(param[0].data.shape),str(param[1].data.shape)
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    for i in range(4):
        feat = net.blobs['conv1'].data[i]
        vis_square(feat)
    for i in range(4):
        feat = net.blobs['pool1'].data[i]
        vis_square(feat)
    for i in range(4):
        feat = net.blobs['conv2'].data[i]
        vis_square(feat)
    for i in range(4):
        feat = net.blobs['pool2'].data[i]
        vis_square(feat)

img.show()

if needShow:
    showStatus()

print "Result:", ''.join(result)

os.remove("image/tmp.jpg")
os.remove("image/part0.jpg")
os.remove("image/part1.jpg")
os.remove("image/part2.jpg")
os.remove("image/part3.jpg")





