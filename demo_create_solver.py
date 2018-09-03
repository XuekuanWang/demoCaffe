from caffe.proto import caffe_pb2

s = caffe_pb2.SolverParameter()

s.train_net = "train.prototxt"
s.test_net.append("test.prototxt")

s.test_interval = 100
s.test_iter.append(10)

s.max_iter = 1000

s.base_lr = 0.1

s.weight_decay = 5e-4

s.lr_policy = "step"

s.display = 10

s.snapshot = 10

s.snapshot_prefix = "model"

s.type = "SGD"

s.solver_mode = caffe_pb2.SolverParameter.GPU

with open("net/s.prototxt", "w") as f:
    f.write(str(s))




