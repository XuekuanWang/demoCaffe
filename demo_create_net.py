import caffe

def create_net():
    net = caffe.NetSpec()

    net.data, net.label = caffe.layers.Data(source="data.lmdb",
                                            backend=caffe.params.Data.LMDB,
                                            batch_size=32,
                                            ntop=2,
                                            transform_param=dict(crop_size=40, mirror=True)
                                            )

    net.conv1 = caffe.layers.Convolution(net.data, num_output=20, kernel_size=5,
                                         weight_filler={"type": "xavier"},
                                         bias_filler={"type":"xavier"})

    net.relu1 = caffe.layers.ReLU(net.conv1, in_place=True)

    net.pool1 = caffe.layers.Pooling(net.relu1, pool=caffe.params.Pooling.MAX,
                                     kernel_size=3, stride=2)

    net.conv2 = caffe.layers.Convolution(net.pool1, num_output=32, kernel_size=3,
                                         pad=1,
                                         weight_filler={"type": "xavier"},
                                         bias_filler={"type": "xavier"})

    net.relu2 = caffe.layers.ReLU(net.conv2, in_place=True)

    net.pool2 = caffe.layers.Pooling(net.relu2, pool=caffe.params.Pooling.MAX,
                                     kernel_size=3, stride=2)

    net.fc3 = caffe.layers.InnerProduct(net.pool2, num_output=1024, weight_filler=dict(type='xavier'))

    net.relu3 = caffe.layers.ReLU(net.fc3, in_place=True)

    ##drop
    net.drop = caffe.layers.Dropout(net.relu3, dropout_param=dict(dropout_ratio=0.5))

    net.fc4 = caffe.layers.InnerProduct(net.drop, num_output=10, weight_filler=dict(type='xavier'))

    net.loss = caffe.layers.SoftmaxWithLoss(net.fc4, net.label)

    with open("net/tt.prototxt", 'w') as f:
        f.write(str(net.to_proto()))


if __name__ == '__main__':
    create_net()