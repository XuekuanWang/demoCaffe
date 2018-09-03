import lmdb
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2

def write():
    # basic setting

    lmdb_file = 'lmdb_data'
    batch_size = 256


    lmdb_env = lmdb.open(lmdb_file, map_size = int(1e12))

    lmdb_txn = lmdb_env.begin(write = True)

    for x in range(batch_size):
        data = np.ones((3, 64, 64), np.uint8)
        label = x

        datum = caffe.io.array_to_datum(data,label)
        keystr = "{:0>8d}".format(x)

        lmdb_txn.put(keystr, datum.SerializeToString())

    lmdb_txn.commit()

def read():
    lmdb_env = lmdb.open('lmdb_data')
    lmdb_txt = lmdb_env.begin()

    datum = caffe_pb2.Datum()

    for key, value in lmdb_txt.cursor():

        datum.ParseFromString(value)

        label = datum.label

        data = caffe.io.datum_to_array(datum)

        print(label)
        print(data)


if __name__ == '__main__':
    write()
    read()