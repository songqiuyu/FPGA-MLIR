# 入口程序：执行量化，生成 QDQ 模型或提取量化参数

import argparse

def main():
    parser = argparse.ArgumentParser(description="COA-MLIR: Model Quantization Tool")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Path to output Quantized model")
    parser.add_argument("--calib_data", type=str, required=True, help="Path to calibration dataset")
    
    args = parser.parse_args()
    
    print(f"[Info] Setting up Quantization Pipeline...")
    print(f"[Info] Input ONNX model: {args.onnx}")
    print(f"[Info] Calibration Data : {args.calib_data}")
    
    # 待实现：调用底层量化接口

if __name__ == "__main__":
    main()
