import os
import sys
import numpy as np
import onnx
import onnxruntime as ort

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from coa_mlir.frontend.interpreter import COAInterpreter

print("=== 开始量化模型对比测试 ===")

# 文件路径
q_onnx = os.path.join(project_root, "models", "onnx_models", "resnet18_quant_int8.onnx")
q_mlir = os.path.join(project_root, "models", "mlir_models", "resnet18_quant_int8.mlir")
q_weight = os.path.join(project_root, "models", "intermediate", "resnet18_quant_int8.npz")

print(f"ONNX 存在: {os.path.exists(q_onnx)}")
print(f"MLIR 存在: {os.path.exists(q_mlir)}")
print(f"Weight 存在: {os.path.exists(q_weight)}")

if not os.path.exists(q_onnx) or not os.path.exists(q_mlir):
    print("缺少文件，跳过测试")
    sys.exit(0)

# 加载模型获取输入名
model = onnx.load(q_onnx)
input_name = model.graph.input[0].name
print(f"输入名: {input_name}")

# 生成随机输入
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# ONNXRuntime 推理
print("\n[1] 运行 ONNXRuntime...")
session = ort.InferenceSession(q_onnx)
ort_result = session.run(None, {input_name: dummy_input})[0]
print(f"ONNXRuntime 输出形状: {ort_result.shape}, dtype: {ort_result.dtype}")

# MLIR 解释器推理
print("\n[2] 运行 MLIR 解释器...")
interpreter = COAInterpreter(q_mlir, q_weight)
input_var_name = input_name.replace(".", "_").replace("/", "_").replace("-", "_")
interpreter.set_input(input_var_name, dummy_input)
mlir_result = interpreter.run()
print(f"MLIR 输出形状: {mlir_result.shape}, dtype: {mlir_result.dtype}")

# 对比结果
print("\n[3] 对比结果...")
# 转换为 float 对比
ort_f = ort_result.astype(np.float32)
mlir_f = mlir_result.astype(np.float32)

max_err = float(np.max(np.abs(ort_f - mlir_f)))
mean_err = float(np.mean(np.abs(ort_f - mlir_f)))

print(f"最大绝对误差: {max_err}")
print(f"平均绝对误差: {mean_err}")

if max_err <= 1.0:
    print("\n[PASS] 测试通过! 量化误差在容忍范围内 (±1)")
else:
    print(f"\n[FAIL] 测试失败! 最大误差 {max_err} 超过容忍阈值 1.0")

print("\n=== Test Complete ===")