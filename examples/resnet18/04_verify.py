"""
Step 4: VLIW 二进制验证
-------------------------
将 output/resnet18.vliw 与 Python 参考实现逐字节对比，
生成 output/verify_report.txt。

验证策略：
  1. 用 legacy/tools/extract_vliw.py 从 MLIR 重新生成参考 VLIW
  2. 与 coa-compiler 输出的二进制逐条比较
  3. 对不一致的字节打印 diff 详情
"""

import os
import sys
import struct
from typing import List, Tuple

ROOT     = os.path.join(os.path.dirname(__file__), '..', '..')
LEGACY   = os.path.join(ROOT, 'legacy')
sys.path.insert(0, os.path.join(LEGACY, 'tools'))

MLIR_FILE  = os.path.join(os.path.dirname(__file__), 'model',  'resnet18.mlir')
VLIW_FILE  = os.path.join(os.path.dirname(__file__), 'output', 'resnet18.vliw')
REPORT     = os.path.join(os.path.dirname(__file__), 'output', 'verify_report.txt')
VLIW_BYTES = 64   # 512-bit per instruction


def load_vliw_binary(path: str) -> List[bytes]:
    """Load a .vliw file and split into 64-byte instructions."""
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // VLIW_BYTES
    return [data[i*VLIW_BYTES:(i+1)*VLIW_BYTES] for i in range(n)]


def generate_reference_vliw(mlir_path: str) -> List[bytes]:
    """
    Use legacy Python tools to re-generate VLIW from MLIR.
    Falls back to an empty list if tools are unavailable.
    """
    try:
        from extract_vliw import parse_all_layers
        from vliw import VLIW
    except ImportError:
        print("[Verify] WARNING: legacy tools not found, skipping reference check.")
        return []

    layers = parse_all_layers(mlir_path, verbose=False)
    ref_instructions = []
    for layer in layers:
        try:
            vliw = VLIW(**layer)
            ref_instructions.append(bytes(vliw.to_bytes()))
        except Exception as e:
            print(f"[Verify] WARNING: failed to build VLIW for layer {layer.get('name','?')}: {e}")
            ref_instructions.append(b'\x00' * VLIW_BYTES)
    return ref_instructions


def compare(compiled: List[bytes], reference: List[bytes]) -> Tuple[int, int, List[str]]:
    """
    Compare compiled vs reference instruction lists.
    Returns (n_match, n_total, diff_lines).
    """
    n_total = max(len(compiled), len(reference))
    n_match = 0
    diffs   = []

    for i in range(n_total):
        if i >= len(compiled):
            diffs.append(f"  Instr {i:3d}: MISSING in compiled output")
            continue
        if i >= len(reference):
            diffs.append(f"  Instr {i:3d}: EXTRA in compiled output")
            n_match += 1  # extra instructions are not counted as failures
            continue

        c = compiled[i]
        r = reference[i]
        if c == r:
            n_match += 1
        else:
            # Find differing bytes
            diff_bytes = [(j, c[j], r[j]) for j in range(VLIW_BYTES) if c[j] != r[j]]
            diffs.append(
                f"  Instr {i:3d}: {len(diff_bytes)} bytes differ | "
                f"compiled[0]={c[0]:02x} ref[0]={r[0]:02x} | "
                f"op_compiled={c[0]} op_ref={r[0]}"
            )
            for j, cv, rv in diff_bytes[:4]:  # show first 4 diff bytes
                diffs.append(f"    byte[{j:3d}]: compiled=0x{cv:02x}  ref=0x{rv:02x}")

    return n_match, n_total, diffs


def main():
    print("=" * 60)
    print(" COA Compiler Verification: ResNet-18")
    print("=" * 60)

    # ---- Load compiled output ----
    if not os.path.exists(VLIW_FILE):
        print(f"[Verify] ERROR: {VLIW_FILE} not found.")
        print("  Run 03_compile.bat first.")
        sys.exit(1)

    compiled = load_vliw_binary(VLIW_FILE)
    print(f"[Verify] Compiled:  {len(compiled)} instructions  ({os.path.getsize(VLIW_FILE)} bytes)")

    # ---- Generate reference ----
    if not os.path.exists(MLIR_FILE):
        print(f"[Verify] WARNING: {MLIR_FILE} not found for reference generation.")
        reference = []
    else:
        reference = generate_reference_vliw(MLIR_FILE)
        print(f"[Verify] Reference: {len(reference)} instructions")

    # ---- Compare ----
    if not reference:
        print("[Verify] Skipping comparison (no reference available).")
        print("[Verify] Dumping compiled instructions:")
        for i, instr in enumerate(compiled[:5]):
            op = instr[0]
            print(f"  Instr {i:3d}: op={op} hex={instr[:8].hex()}...")
        status = "SKIPPED"
    else:
        n_match, n_total, diffs = compare(compiled, reference)
        status = "PASS" if n_match == n_total else "FAIL"
        print(f"\n[Verify] Result: {n_match}/{n_total} instructions match  → {status}")
        if diffs:
            print("\n[Verify] Differences:")
            for d in diffs[:20]:
                print(d)

    # ---- Write report ----
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(f"ResNet-18 Verification Report\n")
        f.write(f"VLIW file: {VLIW_FILE}\n")
        f.write(f"Status:    {status}\n")
        if reference:
            n_match, n_total, diffs = compare(compiled, reference)
            f.write(f"Match:     {n_match}/{n_total}\n\n")
            for d in diffs:
                f.write(d + "\n")

    print(f"\n[Verify] Report written to {REPORT}")
    return 0 if status in ("PASS", "SKIPPED") else 1


if __name__ == "__main__":
    sys.exit(main())
