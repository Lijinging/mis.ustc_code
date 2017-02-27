# coding:utf-8

import caffe
import Image
import requests
import os
import matplotlib.pyplot as plt
import numpy as np

# 获取验证码并转换成灰度图存储
url = "http://mis.teach.ustc.edu.cn/randomImage.do?date='51566564654'"
req = requests.get(url)
table = [0 if i < 120 else i for i in xrange(256)]
try:
    with open("tmp.jpg", 'wb') as file:
        file.write(req.content)
except IOError:
    print "IOError"
finally:
    file.close()
req.close()
img = Image.open("tmp.jpg").convert('L').point(table)
img.crop((00, 0, 20, 20)).save("tmp0.jpg")
img.crop((20, 0, 40, 20)).save("tmp1.jpg")
img.crop((40, 0, 60, 20)).save("tmp2.jpg")
img.crop((60, 0, 80, 20)).save("tmp3.jpg")

model_def = "deploy.prototxt"
model_weights = "misustc_iter.caffemodel"
caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

result = []
labels = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def vis_square(data):  # 显示卷积操作的结果
    """输入一个形如：(n, height, width) or (n, height, width, 3)的数组，
    并对每一个形如(height,width)的特征进行可视化sqrt(n) by sqrt(n)"""

    # 正则化数据
    data = (data - data.min()) / (data.max() - data.min())

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


needShow = True

# 输入数据
for i in [0, 1, 2, 3]:
    im = caffe.io.load_image("tmp" + str(i) + ".jpg", False)
    transformed_image = transformer.preprocess('data', im)
    net.blobs['data'].data[i, ...] = transformed_image
# 进行预测
output = net.forward()

# 显示内部状态
if needShow:
    for i in [0, 1, 2, 3]:
        output_prob = output['prob'][i]
        result.append(labels[output_prob.argmax()])
        print 'predicted class is:', output_prob.argmax()
        print 'output label:', labels[output_prob.argmax()]
        top_inds = output_prob.argsort()[::-1][:5]
        print 'probabilities and labels:', zip(output_prob[top_inds], np.array(list(labels))[top_inds])
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)
    for layer_name, param in net.params.iteritems():
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    for i in [0, 1, 2, 3]:
        feat = net.blobs['conv1'].data[i]
        vis_square(feat)
    for i in [0, 1, 2, 3]:
        feat = net.blobs['conv2'].data[i]
        vis_square(feat)
    for i in [0, 1, 2, 3]:
        feat = net.blobs['pool2'].data[i]
        vis_square(feat)
    for i in [0, 1, 2, 3]:
        feat = net.blobs['ip2'].data[i]
        plt.subplot(2, 1, 1)
        plt.plot(feat.flat)
        plt.subplot(2, 1, 2)
        _ = plt.hist(feat.flat[feat.flat > 0], bins=100)
        plt.show()
    for i in [0, 1, 2, 3]:
        # 分类的聚类结果，峰值对应的标签为预测结果
        feat = net.blobs['prob'].data[i]
        plt.figure(figsize=(15, 3))
        plt.plot(feat.flat)
        plt.show()

img.show()
print result

os.remove("tmp.jpg")
os.remove("tmp0.jpg")
os.remove("tmp1.jpg")
os.remove("tmp2.jpg")
os.remove("tmp3.jpg")
