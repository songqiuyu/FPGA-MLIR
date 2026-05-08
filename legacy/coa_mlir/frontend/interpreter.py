import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import re

class COAInterpreter:
    """
    一个用纯 Python (Numpy) 编写的微型解释器。
    它负责解析生成的 .mlir 文本，并逐层执行对应的数值运算。
    """
    def __init__(self, mlir_path: str, weight_path: str = None):
        self.mlir_path = mlir_path
        self.env = {}
        
        # 挂载权重
        if weight_path:
            weights = np.load(weight_path)
            for k in weights.files:
                var_name = f"%{k.replace('.', '_').replace('/', '_').replace('-', '_')}"
                self.env[var_name] = weights[k]

    def set_input(self, name: str, value: np.ndarray):
        """塞入入口变量（如输入特征图）"""
        self.env[f"%{name}"] = value

    def run(self):
        """执行流"""
        with open(self.mlir_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("module") or line.startswith("func.func") or line.startswith("}"):
                continue
            
            # 返回语句特殊处理
            if line.startswith("return"):
                match = re.search(r'return (.*?):', line)
                if match:
                    ret_vars = [v.strip() for v in match.group(1).split(",")]
                    return self.env[ret_vars[0]]
                
            # 匹配 MLIR 行模式: %out = "coa.op"(%in1, %in2) { attr = value } : ()
            match = re.search(r'(\%[\w_]+)\s*=\s*"coa\.(\w+)"\((.*?)\)(?:\s*\{(.*?)\})?', line)
            if match:
                out_var = match.group(1)
                op_type = match.group(2)
                inputs_str = match.group(3).split(",")
                inputs = [i.strip() for i in inputs_str if i.strip()]
                attrs_str = match.group(4)
                
                # 解析属性
                attrs = {}
                if attrs_str:
                    # 替换掉 "=" 变为 ":" 加上外层的 "{}" 使其成为合法的 python dict 字符串
                    # 但注意原来的键没有引号，所以我们要用正则给键加上引号
                    dict_str = "{" + attrs_str.replace("=", ":") + "}"
                    dict_str = re.sub(r'([a-zA-Z_]\w*)\s*:', r'"\1":', dict_str)
                    import ast
                    try:
                        attrs = ast.literal_eval(dict_str)
                    except Exception as e:
                        print(f"Failed to parse attributes: {attrs_str}")
                        print(e)
                
                # 执行算子并保存返回值到环境变量桶
                self.env[out_var] = self._exec_op(op_type, inputs, attrs)

    def _exec_op(self, op_type, inputs, attrs):
        in_tensors = [self.env[i] for i in inputs]
        
        # --- 原始浮点指令 ---
        if op_type == "conv":
            X, W = in_tensors[0], in_tensors[1]
            B = in_tensors[2] if len(in_tensors) > 2 else None
            return self._op_conv2d(X, W, B, attrs)
        elif op_type == "relu":
            return np.maximum(in_tensors[0], 0)
        elif op_type == "add":
            return in_tensors[0] + in_tensors[1]
        elif op_type == "maxpool":
            return self._op_maxpool(in_tensors[0], attrs)
        elif op_type == "globalaveragepool":
            return np.mean(in_tensors[0], axis=(2, 3), keepdims=True)
        elif op_type == "flatten":
            return in_tensors[0].reshape(in_tensors[0].shape[0], -1)
        elif op_type == "gemm":
            X, W = in_tensors[0], in_tensors[1]
            B = in_tensors[2] if len(in_tensors) > 2 else None
            if attrs.get('transB', 0): W = W.T
            out = np.dot(X, W)
            if B is not None: out += B
            return out
            
        # --- 量化强相关指令 ---
        elif op_type == "quantizelinear":
            scale, zp = attrs.get('scale', 1.0), attrs.get('zp', 0)
            X = in_tensors[0]
            out = np.round(X / scale) + zp
            return np.clip(out, -128, 127).astype(np.int8)
            
        elif op_type == "dequantizelinear":
            scale, zp = attrs.get('scale', 1.0), attrs.get('zp', 0)
            X = in_tensors[0]
            # ONNXRuntime中的 Dequantize 是减去ZP后乘Scale恢复浮点
            return (X.astype(np.float32) - zp) * scale
            
        elif op_type == "qlinearconv":
            X, W = in_tensors[0], in_tensors[1]
            B = in_tensors[2] if len(in_tensors) > 2 else None
            in_s, in_z = attrs["in_scale"], attrs["in_zp"]
            w_s, w_z = attrs["weight_scale"], attrs["weight_zp"]
            out_s, out_z = attrs["out_scale"], attrs["out_zp"]
            
            x_f = X.astype(np.float32) - in_z
            w_f = W.astype(np.float32)
            if isinstance(w_z, list): w_f -= np.array(w_z).reshape(-1, 1, 1, 1)
            else: w_f -= w_z
                
            acc = self._op_conv2d(x_f, w_f, B, attrs)
            M = (in_s * np.array(w_s)).reshape(1, -1, 1, 1) / out_s if isinstance(w_s, list) else (in_s * w_s) / out_s 
            out = np.round(acc * M) + out_z
            return np.clip(out, -128, 127).astype(np.int8)
            
        elif op_type == "qlinearadd":
            A, B = in_tensors[0], in_tensors[1]
            a_s, a_z = attrs["a_scale"], attrs["a_zp"]
            b_s, b_z = attrs["b_scale"], attrs["b_zp"]
            c_s, c_z = attrs["out_scale"], attrs["out_zp"]
            
            a_f = (A.astype(np.float32) - a_z) * a_s
            b_f = (B.astype(np.float32) - b_z) * b_s
            c_f = a_f + b_f
            out = np.round(c_f / c_s) + c_z
            return np.clip(out, -128, 127).astype(np.int8)
            
        elif op_type == "qlinearglobalaveragepool":
            X = in_tensors[0]
            in_s, in_z = attrs["in_scale"], attrs["in_zp"]
            out_s, out_z = attrs["out_scale"], attrs["out_zp"]
            x_f = (X.astype(np.float32) - in_z) * in_s
            pool_f = np.mean(x_f, axis=(2, 3), keepdims=True)
            out = np.round(pool_f / out_s) + out_z
            return np.clip(out, -128, 127).astype(np.int8)
            
        elif op_type == "qgemm":
            A, B_mat = in_tensors[0], in_tensors[1]
            C = in_tensors[2] if len(in_tensors) > 2 else None
            a_s, a_z = attrs["a_scale"], attrs["a_zp"]
            b_s, b_z = attrs["b_scale"], attrs["b_zp"]
            out_s, out_z = attrs["out_scale"], attrs["out_zp"]
            
            a_f = A.astype(np.float32) - a_z
            b_f = B_mat.astype(np.float32) - b_z
            if attrs.get("transB", 0): b_f = b_f.T
            
            acc = np.dot(a_f, b_f) * attrs.get("alpha", 1.0)
            if C is not None: acc += C.astype(np.float32)
            M = (a_s * b_s) / out_s
            out = np.round(acc * M) + out_z
            return np.clip(out, -128, 127).astype(np.int8)

        else:
            raise NotImplementedError(f"Unsupported coa.op: {op_type}")

    def _op_conv2d(self, X, W, B, attrs):
        # 纯 Numpy 的高效版滑动窗口 Conv2D 计算
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])  # beg_h, beg_w, end_h, end_w
        
        N, C, H, W_in = X.shape
        M, C_w, kH, kW = W.shape
        
        X_pad = np.pad(X, ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])), mode='constant')
        out_H = (H + pads[0] + pads[2] - kH) // strides[0] + 1
        out_W = (W_in + pads[1] + pads[3] - kW) // strides[1] + 1
        
        # 利用内存视图(sliding_window_view)避免显式的 Python 长时间低效 for 循环
        windows = sliding_window_view(X_pad, (1, C, kH, kW))
        windows = windows[:, 0, ::strides[0], ::strides[1], 0, :, :, :]
        # einsum 代替张量乘法，加快执行
        out = np.einsum('nhwckl,mckl->nmhw', windows, W, optimize=True)
        
        if B is not None:
            out += B.reshape(1, M, 1, 1)
        return out

    def _op_maxpool(self, X, attrs):
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        kernel_shape = attrs.get("kernel_shape", [3, 3])
        kH, kW = kernel_shape
        
        if X.dtype == np.uint8: pad_val = 0
        elif X.dtype == np.int8: pad_val = -128
        else: pad_val = -np.inf
            
        X_pad = np.pad(X, ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])), mode='constant', constant_values=pad_val)
        N, C, H_pad, W_pad = X_pad.shape
        
        out_H = (X.shape[2] + pads[0] + pads[2] - kH) // strides[0] + 1
        out_W = (X.shape[3] + pads[1] + pads[3] - kW) // strides[1] + 1
        
        windows = sliding_window_view(X_pad, (1, 1, kH, kW))
        windows = windows[:, :, ::strides[0], ::strides[1], 0, 0, :, :]
        out = np.max(windows, axis=(-2, -1))
        return out
