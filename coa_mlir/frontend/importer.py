import onnx
from onnx import numpy_helper
import numpy as np
import os

class ONNXImporter:
    """
    负责从指定路径加载 ONNX 模型，并解析出底层的计算图(Graph)。
    同时负责将所有的权重参数(Initializer)抽取出来保存到中间文件。
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        print(f"[Importer] 正在从 {self.model_path} 加载 ONNX 模型...")
        try:
            self.model = onnx.load(self.model_path)
            print(f"[Importer] 成功加载模型! IR 版号: {self.model.ir_version}, 算子集(Opset): {self.model.opset_import[0].version}")
            return self.model.graph
        except Exception as e:
            print(f"[Importer Error] 加载 ONNX 模型失败: {e}")
            raise e
            
    def extract_weights(self, output_dir: str):
        """
        把图中的静态权重(Initializer)全部抽取出来，存成字典并保存到磁盘。
        """
        print("[Importer] 开始抽取模型权重 (Initializers)...")
        if not self.model:
            raise ValueError("请先调用 load() 加载模型！")
            
        weights = {}
        for init in self.model.graph.initializer:
            # 使用 onnx 自带工具直接反序列化为 numpy 数组
            weights[init.name] = numpy_helper.to_array(init)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 将它保存为 npz 格式文件
        base_name = os.path.basename(self.model_path).replace(".onnx", ".npz")
        save_path = os.path.join(output_dir, base_name)
        np.savez(save_path, **weights)
        print(f"[Importer] 成功导出 {len(weights)} 个权重张量到 {save_path}")
        return weights
