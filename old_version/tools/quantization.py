# 入口程序：执行量化，生成特定格式的量化模型

import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from coa_mlir.quantization.calibrator import NpyCalibrationDataReader

def main():
    parser = argparse.ArgumentParser(description="COA-MLIR: Model Quantization Tool")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Path to output Quantized model")
    parser.add_argument("--calib_data", type=str, required=True, help="Path to calibration dataset (.npy folder)")
    parser.add_argument("--activation_type", type=str, choices=["int8", "uint8"], default="int8", help="Activation quantization type (default: int8)")
    
    args = parser.parse_args()
    
    print("=========================================")
    print("[COA-MLIR] Static Quantization Process   ")
    print("=========================================")
    print(f"Input model : {args.onnx}")
    print(f"Calib Data  : {args.calib_data}")
    print(f"Output model: {args.output}")
    print(f"Act Type    : {args.activation_type} (Weight fixed to int8)\n")
    
    try:
        # 1. 实例化专门给 numpy 文件阵列准备的读取器
        dr = NpyCalibrationDataReader(args.calib_data, args.onnx)
        
        print("[Quantizer] 开始搜集校准数据激活最大最小值分布...")
        
        act_quant_type = QuantType.QUInt8 if args.activation_type == "uint8" else QuantType.QInt8
        
        # 2. 执行核心静态量化
        quantize_static(
            model_input=args.onnx,
            model_output=args.output,
            calibration_data_reader=dr,
            quant_format=QuantFormat.QOperator,  # 采用 QOperator 格式
            activation_type=act_quant_type,      # 支持灵活切换的激活值类型
            weight_type=QuantType.QInt8          # 权重固定采取 INT8
        )
        print(f"\n[Success] 静态量化顺利完成! 新模型已落盘至: {args.output}")
        
    except Exception as e:
        print(f"\n[Error] 量化执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
