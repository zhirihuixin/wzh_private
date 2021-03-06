from tensorflow.python import pywrap_tensorflow
import cPickle

reader = pywrap_tensorflow.NewCheckpointReader("inception_v1.ckpt")
var_to_shape_map = reader.get_variable_to_shape_map()

result = [[[] for _ in xrange(2)] for _ in xrange(len(var_to_shape_map))]
i = 0
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(type(reader.get_tensor(key))) # Remove this is you want to print only variable names
    result[i][0] = key
    result[i][1] = reader.get_tensor(key)
    i = i + 1

with open('inception_v1.pkl', 'wb') as f:
    cPickle.dump(result, f, cPickle.HIGHEST_PROTOCOL)

import cPickle

with open('inception_v1.pkl', 'rb') as f:
    data = cPickle.load(f)
print len(data)
for i in xrange(len(data)):
    print data[i][0]
    print data[i][1].shape

import h5py
import cPickle
import copy

f = h5py.File('D:\wangzhihui\inception\\xception\\xception_weights_tf_dim_ordering_tf_kernels.h5')
result = []
a = [[], []]
def printname(name):
    print name
    if ':0' in name:
        a[0] = name
        a[1] = f[name][:]
        result.append(copy.deepcopy(a))
f.visit(printname)

with open('xception.pkl', 'wb') as f:
    cPickle.dump(result, f, cPickle.HIGHEST_PROTOCOL)
