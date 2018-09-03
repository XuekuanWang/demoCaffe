import sys
sys.path.append('/home/kuan/AM-softmax_caffe/python')
import caffe
import numpy as np

##caffemodel deploy.prototxt

deploy = "/home/kuan/PycharmProjects/demo_cnn_net/cnn_net/alexnet/deploy.prototxt"

model = "/home/kuan/PycharmProjects/demo_cnn_net/cnn_model/cifar/alexnet/alexnet_iter_110.caffemodel"

net = caffe.Net(deploy, model, caffe.TEST)


net.blobs["data"].data[...] = np.ones((3,32,32),np.uint8)

net.forward()

prob = net.blobs["prob"].data[0]

print(prob)

