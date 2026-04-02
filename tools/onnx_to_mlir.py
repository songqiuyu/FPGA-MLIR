# 入口程序：将 ONNX 模型转换为 MLIR

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="COA-MLIR: ONNX to MLIR Transpiler")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--mlir", type=str, required=True, help="Path to output MLIR file")
    
    args = parser.parse_args()
    
    print(f"[Info] Setting up ONNX to MLIR conversion...")
    print(f"[Info] Input ONNX model: {args.onnx}")
    print(f"[Info] Output MLIR file: {args.mlir}")
    
    # 待实现：调用底层转译接口

if __name__ == "__main__":
    main()
