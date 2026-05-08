import numpy as np
import os

npz_path = "d:/PROJECT/FPGA-MLIR/models/intermediate/resnet18_quant_int8.npz"
npz = np.load(npz_path)

print("Keys in npz file:")
for k in sorted(npz.files):
    print(f"  {k}: {npz[k].shape}")