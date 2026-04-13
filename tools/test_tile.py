"""测试 calculate_buffer_consumption - 简单版"""
import numpy as np

def calculate_buffer_consumption(tN, tM, tR, tC, N, M, kernel, stride, pad, dilation):
    """计算buffer使用情况，返回0表示OK"""
    # tN32_rd = ceil(tN/32) 的数量
    tN32_rd = 0
    for sn in range(0, N, tN):
        t = ((sn + tN + 31) >> 5) - (sn >> 5)
        if t > tN32_rd:
            tN32_rd = t

    tM32_rd = 0
    for sm in range(0, M, tM):
        t = ((sm + tM + 15) >> 4) - (sm >> 4)
        if t > tM32_rd:
            tM32_rd = t

    relems = (tR - 1) * stride + (kernel - 1) * dilation + 1
    celems = (tC - 1) * stride + (kernel - 1) * dilation + 1
    gdepth = relems * celems * tN32_rd
    wbuf_single_m_size = tN32_rd * kernel * kernel
    wdepth = wbuf_single_m_size * tM32_rd
    odepth = tR * tC * tM32_rd

    print(f"tN={tN}, tM={tM}, tR={tR}, tC={tC}, kernel={kernel}, stride={stride}")
    print(f"  tN32_rd={tN32_rd}, tM32_rd={tM32_rd}")
    print(f"  relems={relems}, celems={celems}")
    print(f"  gdepth={gdepth}, wdepth={wdepth}, odepth={odepth}")

    if wdepth >= 256:
        print(f"  -> flag=1 (wdepth >= 256)")
        return 1
    if gdepth >= 1024:
        print(f"  -> flag=2 (gdepth >= 1024)")
        return 2
    if odepth >= 2048:
        print(f"  -> flag=3 (odepth >= 2048)")
        return 3
    print(f"  -> flag=0 (OK)")
    return 0

# 第一层卷积参数
N = 3
M = 64
R = 112
C = 112
kernel = 7
stride = 2
pad = 3
dilation = 1

print("=" * 60)
print(f"Testing: N={N}, M={M}, R={R}, C={C}, kernel={kernel}, stride={stride}, pad={pad}")
print("=" * 60)

tN = N
tM = M
tR = R
tC = C

flag = calculate_buffer_consumption(tN, tM, tR, tC, N, M, kernel, stride, pad, dilation)