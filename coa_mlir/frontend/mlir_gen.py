import onnx
from onnx import AttributeProto

class MLIRTextBuilder:
    """
    负责将 ONNX Graph 转换为基础的 MLIR 文本表示。
    """
    def __init__(self, graph):
        self.graph = graph

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
            # MLIR 中变量名称需要规范，替换掉不支持的字符
            name = v.name.replace(".", "_").replace("/", "_").replace("-", "_")
            
            # 简单的类型推导，如果没有 shape 则回退到 *xf32
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
            
            # 解析属性 (Attributes)
            attr_dict = {}
            for attr in node.attribute:
                attr_dict[attr.name] = self._parse_attr_value(attr)
            
            attr_str = ""
            if attr_dict:
                attr_items = [f'{k} = {v}' for k, v in attr_dict.items()]
                attr_str = " { " + ", ".join(attr_items) + " }"
            
            # 获取输入和输出，规范化命名
            ins = [f"%{i.replace('.', '_').replace('/', '_').replace('-', '_')}" for i in node.input if i]
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
