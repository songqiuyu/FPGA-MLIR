import numpy as np

output0 = np.fromfile('_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_output_0')
output1 = np.fromfile('_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_output_0')
output2 = np.fromfile('_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_output_0')
output3 = np.fromfile('_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_output_0')
output4 = np.fromfile('_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_output_0')
output5 = np.fromfile('_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_output_0')

res = np.concatenate([output0, output1, output2, output3, output4, output5])
res.tofile('result.bin')