import os
import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader

class NpyCalibrationDataReader(CalibrationDataReader):
    """
    提供给 ONNXRuntime 的符合规范的校准数据采集器。
    它会自动遍历指定的文件夹从中读取 .npy 文件作为输入流供底层执行正向推理。
    """
    def __init__(self, data_dir: str, model_path: str):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.enum_data = iter(self.file_names)
        
        # 自动探测模型的第一个节点名称并作为给定的字典 Key，因为 resnet18 的首节点名不一定叫 input
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        
    def get_next(self):
        try:
            file_name = next(self.enum_data)
        except StopIteration:
            return None
            
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path)
        
        # ONNXRuntime 要求每次迭代返回一个包含该批次所有输入的 Dictionary
        return {self.input_name: data}

    def rewind(self):
        self.enum_data = iter(self.file_names)
