"""
coa.tiling — Hardware-aware tiling constraint checker and tile search.

Python port of compiler/lib/Transforms/Tiling.cpp (checkBuffers / getTile).

Buffer constraints (default limits match --coa-tiling pass defaults):
  wdepth = ceil(tN/32) * k*k * ceil(tM/16)          < WDEPTH_LIMIT (256)
  gdepth = ((tR-1)*s+(k-1)*d+1) * (...tC...) * ceil(tN/32)  < GDEPTH_LIMIT (1024)
  odepth = tR * tC * ceil(tM/16)                              < ODEPTH_LIMIT (2048)
"""

from typing import Tuple

WDEPTH_LIMIT: int = 256
GDEPTH_LIMIT: int = 1024
ODEPTH_LIMIT: int = 2048


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tN32_max(N: int, tN: int) -> int:
    """Maximum number of 32-wide input-channel groups touched by one tN strip."""
    if N <= 0 or tN <= 0:
        return 1
    return max(_ceil_div(sn + tN, 32) - sn // 32 for sn in range(0, N, tN))


def _tM16_max(M: int, tM: int) -> int:
    """Maximum number of 16-wide output-channel groups touched by one tM strip."""
    if M <= 0 or tM <= 0:
        return 1
    return max(_ceil_div(sm + tM, 16) - sm // 16 for sm in range(0, M, tM))


def calculate_buffer_consumption(
    tN: int, tM: int, tR: int, tC: int,
    N: int, M: int, k: int, s: int, pad: int, d: int,
    wlim: int = WDEPTH_LIMIT,
    glim: int = GDEPTH_LIMIT,
    olim: int = ODEPTH_LIMIT,
) -> int:
    """
    Check hardware buffer constraints.

    Returns:
        0  — all constraints satisfied (tile is legal)
        1  — wdepth over limit
        2  — gdepth over limit
        3  — odepth over limit

    Matches C++ checkBuffers() in Tiling.cpp exactly.
    """
    tN32 = _tN32_max(N, tN)
    tM16 = _tM16_max(M, tM)

    relems = (tR - 1) * s + (k - 1) * d + 1
    celems = (tC - 1) * s + (k - 1) * d + 1
    wdepth = tN32 * k * k * tM16
    gdepth = relems * celems * tN32
    odepth = tR * tC * tM16

    if wdepth >= wlim:
        return 1
    if gdepth >= glim:
        return 2
    if odepth >= olim:
        return 3
    return 0


def get_tile(
    N: int, M: int, R: int, C: int,
    k: int, s: int, pad: int = 0, d: int = 1,
    wlim: int = WDEPTH_LIMIT,
    glim: int = GDEPTH_LIMIT,
    olim: int = ODEPTH_LIMIT,
) -> Tuple[int, int, int]:
    """
    Find the largest legal tile (tM, tR, tC) using a greedy shrink-then-grow
    heuristic.  Matches C++ getTile() in Tiling.cpp exactly.

    Returns (tM, tR, tC).
    """
    tM, tR, tC = M, R, C

    def _flag() -> int:
        return calculate_buffer_consumption(N, tM, tR, tC, N, M, k, s, pad, d, wlim, glim, olim)

    f = _flag()
    while f != 0:
        if f == 1:
            # wdepth violation — shrink tM
            if   tM % 2 == 0: tM //= 2
            elif tM % 3 == 0: tM //= 3
            elif tM % 5 == 0: tM //= 5
            else:              tM = 16
            if tM < 16:
                tM = 16
        else:
            # gdepth / odepth violation — shrink tR first, then tC
            if tR != 1:
                if   tR % 2 == 0: tR //= 2
                elif tR % 3 == 0: tR //= 3
                else:              tR = 1
            elif tC != 1:
                if   tC % 2 == 0: tC //= 2
                elif tC % 3 == 0: tC //= 3
                else:              tC = 1
            else:
                break   # cannot reduce further
        if tM < 16:
            tM = 16
        f = _flag()

    # Greedy enlarge tR (try successive integer multiples up to R)
    scale = 2
    while scale * tR <= R:
        new_tR = scale * tR
        if calculate_buffer_consumption(N, tM, new_tR, tC, N, M, k, s, pad, d, wlim, glim, olim) == 0:
            tR = new_tR
        scale += 1

    return tM, tR, tC


def buffer_utilization(
    tN: int, tM: int, tR: int, tC: int,
    N: int, M: int, k: int, s: int, d: int,
    wlim: int = WDEPTH_LIMIT,
    glim: int = GDEPTH_LIMIT,
    olim: int = ODEPTH_LIMIT,
) -> Tuple[float, float, float]:
    """
    Return (wutil, gutil, outil) in [0, 1]; a value > 1.0 means over-limit.
    Used by the RL/Bayesian tiling optimizers as a continuous reward signal.
    """
    tN32 = _tN32_max(N, tN)
    tM16 = _tM16_max(M, tM)

    relems = (tR - 1) * s + (k - 1) * d + 1
    celems = (tC - 1) * s + (k - 1) * d + 1
    wdepth = tN32 * k * k * tM16
    gdepth = relems * celems * tN32
    odepth = tR * tC * tM16

    return wdepth / wlim, gdepth / glim, odepth / olim
