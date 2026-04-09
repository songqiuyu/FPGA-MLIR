import numpy as np


weight_name = 'model.2.cv1.conv.weight_quantized'
bias_name = 'model.2.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(64, 64, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(64)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')

weight_name = 'model.4.cv1.conv.weight_quantized'
bias_name = 'model.4.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(128, 128, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(128)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')

weight_name = 'model.6.cv1.conv.weight_quantized'
bias_name = 'model.6.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(256, 256, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(256)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')

weight_name = 'model.8.cv1.conv.weight_quantized'
bias_name = 'model.8.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(512, 512, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(512)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')



weight_name = 'model.10.cv1.conv.weight_quantized'
bias_name = 'model.10.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(512, 512, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(512)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')


weight_name = 'model.13.cv1.conv.weight_quantized'
bias_name = 'model.13.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(256, 768, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(256)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')


weight_name = 'model.16.cv1.conv.weight_quantized'
bias_name = 'model.16.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(128, 384, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(128)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')

weight_name = 'model.19.cv1.conv.weight_quantized'
bias_name = 'model.19.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(256, 384, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(256)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')

weight_name = 'model.22.cv1.conv.weight_quantized'
bias_name = 'model.22.cv1.conv.bias_quantized'


weight = np.fromfile(weight_name, dtype=np.int8).reshape(512, 768, 1, 1)
result = np.split(weight, 2, axis=0)
result[0].tofile(weight_name + '_split_0')
result[1].tofile(weight_name + '_split_1')

bias = np.fromfile(bias_name, dtype=np.int32).reshape(512)
result = np.split(bias, 2)
result[0].tofile(bias_name + '_split_0')
result[1].tofile(bias_name + '_split_1')