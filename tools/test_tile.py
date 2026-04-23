"""测试 tile 计算 - 修正版"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from assign_addr import get_tile

# C hex 实际值
c_hex = [
    (64, 1, 112),  # Layer 1
    (64, 4, 56),  # Layer 2
    (64, 1, 56),  # Layer 3
    (64, 4, 56),  # Layer 4
    (64, 1, 56),  # Layer 5
    (128, 2, 28), # Layer 6
    (128, 4, 28), # Layer 7
]

# 测试用例
tests = [
    (3, 64, 112, 112, 7, 2, 3),
    (64, 64, 56, 56, 3, 1, 1),
    (64, 64, 56, 56, 3, 1, 1),
    (64, 64, 56, 56, 3, 1, 1),
    (64, 64, 56, 56, 3, 1, 1),
    (64, 128, 28, 28, 3, 2, 1),
    (64, 128, 28, 28, 3, 2, 1),
]

print("Testing get_tile:")
for i, (N,M,R,C,k,s,p) in enumerate(tests):
    tM, tR, tC = get_tile(N, M, R, C, k, s, p, 1)
    exp = c_hex[i]
    match = "OK" if (tM,tR,tC) == exp else "FAIL"
    print(f"Layer {i+1}: ({tM},{tR},{tC}) expected={exp} {match}")