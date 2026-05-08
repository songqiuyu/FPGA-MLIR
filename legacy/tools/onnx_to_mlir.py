import argparse
import sys
import os

# 将根目录加到 sys.path 以便能引用 coa_mlir 包
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from coa_mlir.frontend.importer import ONNXImporter
from coa_mlir.frontend.mlir_gen import MLIRTextBuilder

def main():
    parser = argparse.ArgumentParser(description="COA-MLIR: ONNX to MLIR Transpiler")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument("--mlir", type=str, required=True, help="Path to output MLIR file")
    
    args = parser.parse_args()
    
    print("=========================================")
    print("[COA-MLIR] ONNX to MLIR Translation Start")
    print("=========================================")
    print(f"Input model : {args.onnx}")
    print(f"Output MLIR : {args.mlir}\n")
    
    try:
        # 第一步：导入模型并解析得到 Graph
        importer = ONNXImporter(args.onnx)
        graph = importer.load()
        
        # [新增] 并行步骤：抽取模型参数并保存到 intermediate 目录
        intermediate_dir = os.path.join(project_root, "models", "intermediate")
        weights_dict = importer.extract_weights(intermediate_dir)
        
        # 第二步：将 Graph 内容翻译为 MLIR 文本格式
        builder = MLIRTextBuilder(graph, weights_dict)
        mlir_text = builder.generate()
        
        # 第三步：输出文件
        with open(args.mlir, "w", encoding="utf-8") as f:
            f.write(mlir_text)
            
        print(f"\n[Success] MLIR representation successfully written to {args.mlir}")
    
    except Exception as e:
        print(f"\n[Error] Translation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
