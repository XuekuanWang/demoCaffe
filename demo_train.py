import sys
sys.path.append('/home/kuan/AM-softmax_caffe/python')
import caffe

solver = caffe.SGDSolver("/home/kuan/PycharmProjects/demo_cnn_net/cnn_net/alexnet/solver.prototxt")

solver.solve()