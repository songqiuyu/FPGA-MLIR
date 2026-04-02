import onnx
from onnx import AttributeProto

class MLIRTextBuilder:
    """
    负责将 ONNX Graph 转换为基础的 MLIR 文本表示。
    引入了权重量化字典的支持，能够在转译时降维抽取参数放入 MLIR Attributes 中。
    """
    def __init__(self, graph, weights_dict=None):
        self.graph = graph
        self.weights = weights_dict or {}

    def _get_val(self, tensor_name):
        """尝试从持久化字典中取出浮点或纯数值张量"""
        if tensor_name in self.weights:
            arr = self.weights[tensor_name]
            # 如果是只有一维的一个标量元素，直接转为标准Python数字
            if arr.size == 1:
                val = arr.item()
                return int(val) if isinstance(val, int) or str(arr.dtype).startswith('int') or str(arr.dtype).startswith('uint') else float(val)
            else:
                # 否则给出去精度截断的浮点数组以便放入 MLIR Attribute
                return [round(float(x), 6) for x in arr.flatten()]
        return None

    def _parse_attr_value(self, attr):
        if attr.type == AttributeProto.INT:
            return str(attr.i)
        elif attr.type == AttributeProto.FLOAT:
            return f"{attr.f:.4f}"
        elif attr.type == AttributeProto.INTS:
            return "[" + ", ".join(str(v) for v in attr.ints) + "]"
        elif attr.type == AttributeProto.FLOATS:
            return "[" + ", ".join(str(v) for v in attr.floats) + "]"
        elif attr.type == AttributeProto.STRING:
            return f'"{attr.s.decode("utf-8")}"'
        else:
            return f'"<Unsupported_Attr_{attr.type}>"'

    def generate(self) -> str:
        lines = []
        lines.append("module {")
        
        # 1. 提取函数的输入参数 (Inputs) 作为 func 的签名
        func_args = []
        for v in self.graph.input:
            name = v.name.replace(".", "_").replace("/", "_").replace("-", "_")
            shape_str = "*xf32"
            try:
                dims = [str(d.dim_value) if d.dim_value > 0 else "?" for d in v.type.tensor_type.shape.dim]
                shape_str = "x".join(dims) + "xf32" if dims else "*xf32"
            except:
                pass
            func_args.append(f"%{name}: tensor<{shape_str}>")
            
        args_str = ", ".join(func_args)
        lines.append(f"  func.func @main({args_str}) {{")

        # 2. 遍历所有的计算节点
        for node in self.graph.node:
            op_type = node.op_type.lower()
            
            # 解析自身携带的属性 (Attributes)
            attr_dict = {}
            for attr in node.attribute:
                attr_dict[attr.name] = self._parse_attr_value(attr)
            
            # --- 量化算子降维特判 (提取 S 与 ZP 放入 Attributes) ---
            core_ins = []
            if op_type == "qlinearconv":
                core_ins = [node.input[0], node.input[3]]
                if len(node.input) > 8 and node.input[8]: core_ins.append(node.input[8])
                attr_dict["in_scale"] = self._get_val(node.input[1])
                attr_dict["in_zp"] = self._get_val(node.input[2])
                attr_dict["weight_scale"] = self._get_val(node.input[4])
                attr_dict["weight_zp"] = self._get_val(node.input[5])
                attr_dict["out_scale"] = self._get_val(node.input[6])
                attr_dict["out_zp"] = self._get_val(node.input[7])
            elif op_type in ["quantizelinear", "dequantizelinear"]:
                core_ins = [node.input[0]]
                attr_dict["scale"] = self._get_val(node.input[1])
                attr_dict["zp"] = self._get_val(node.input[2])    
            elif op_type == "qlinearadd":
                core_ins = [node.input[0], node.input[3]]
                attr_dict["a_scale"] = self._get_val(node.input[1])
                attr_dict["a_zp"] = self._get_val(node.input[2])
                attr_dict["b_scale"] = self._get_val(node.input[4])
                attr_dict["b_zp"] = self._get_val(node.input[5])
                attr_dict["out_scale"] = self._get_val(node.input[6])
                attr_dict["out_zp"] = self._get_val(node.input[7])
            elif op_type == "qlinearglobalaveragepool":
                core_ins = [node.input[0]]
                attr_dict["in_scale"] = self._get_val(node.input[1])
                attr_dict["in_zp"] = self._get_val(node.input[2])
                attr_dict["out_scale"] = self._get_val(node.input[3])
                attr_dict["out_zp"] = self._get_val(node.input[4])
            elif op_type == "qgemm":
                core_ins = [node.input[0], node.input[3]]
                if len(node.input) > 6 and node.input[6]: core_ins.append(node.input[6])
                attr_dict["a_scale"] = self._get_val(node.input[1])
                attr_dict["a_zp"] = self._get_val(node.input[2])
                attr_dict["b_scale"] = self._get_val(node.input[4])
                attr_dict["b_zp"] = self._get_val(node.input[5])
                if len(node.input) > 8:
                    attr_dict["out_scale"] = self._get_val(node.input[7])
                    attr_dict["out_zp"] = self._get_val(node.input[8])
            else:
                # 普通算子直接接客所有 input
                core_ins = [i for i in node.input if i]

            # 组装格式化字符串
            attr_str = ""
            if attr_dict:
                attr_items = []
                for k, v in attr_dict.items():
                    if isinstance(v, list) and not isinstance(v, str):
                        # 过长的数组就截断省略, 但是这里为了真实反映量化的 channel 特性尽量全印，或者打印成 [v0, v1...]
                        v_str = "[" + ", ".join(map(str, v)) + "]"
                        attr_items.append(f'{k} = {v_str}')
                    else:
                        attr_items.append(f'{k} = {v}')
                attr_str = " { " + ", ".join(attr_items) + " }"
            
            # 获取输入和输出，规范化命名
            ins = [f"%{i.replace('.', '_').replace('/', '_').replace('-', '_')}" for i in core_ins]
            outs = [f"%{o.replace('.', '_').replace('/', '_').replace('-', '_')}" for o in node.output if o]
            
            ins_str = ", ".join(ins)
            outs_str = ", ".join(outs)
            
            # 构造成自定义方言的形态并附着属性
            if outputs_str := outs_str:
                lines.append(f'    {outputs_str} = "coa.{op_type}"({ins_str}){attr_str} : ()')
            else:
                lines.append(f'    "coa.{op_type}"({ins_str}){attr_str} : ()')

        # 3. 添加 return 语句
        ret_outs = [f"%{o.name.replace('.', '_').replace('/', '_').replace('-', '_')}" for o in self.graph.output]
        lines.append(f'    return {", ".join(ret_outs)} : ()')
        lines.append("  }")
        lines.append("}")
        
        return "\n".join(lines)
