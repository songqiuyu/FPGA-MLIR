import unittest
import os
import sys

# 将根目录加到 sys.path 以便测试框架能够导入 coa_mlir 包
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from coa_mlir.frontend.importer import ONNXImporter

class TestFrontend(unittest.TestCase):
    def setUp(self):
        # 配置测试的 ONNX 模型路径，默认使用我们在工作区里存在的 resnet18.onnx
        self.test_model_path = os.path.join(project_root, "models", "onnx_models", "resnet18.onnx")

    def test_importer_load(self):
        """测试 ONNXImporter 是否能正常的加载模型解析图结构"""
        if not os.path.exists(self.test_model_path):
            self.skipTest("Test skipped: resnet18.onnx not found in the designated folder.")
        
        importer = ONNXImporter(self.test_model_path)
        graph = importer.load()
        
        self.assertIsNotNone(graph, "加载返回得到的 graph 不应为空")
        self.assertTrue(len(graph.node) > 0, "解析后的计算图中应该包含计算节点")

if __name__ == "__main__":
    unittest.main()
