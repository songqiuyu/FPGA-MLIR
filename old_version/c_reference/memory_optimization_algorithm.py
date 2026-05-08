#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图着色算法的神经网络推理内存地址优化
Memory Address Optimization for Neural Network Inference using Graph Coloring Algorithm

核心思想：
1. 将每个tensor的生命周期建模为图中的节点
2. 如果两个tensor的生命周期重叠，则它们之间有边连接
3. 使用图着色算法为每个"颜色"（内存块）分配地址
4. 同一颜色的tensor可以共享内存空间，因为它们的生命周期不重叠

作者: AI Assistant
日期: 2024年10月
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import networkx as nx
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class TensorInfo:
    """张量信息"""
    name: str
    size: int  # 张量大小（字节）
    birth_time: int  # 创建时间
    death_time: int  # 销毁时间
    
    def lifetime_overlaps(self, other: 'TensorInfo') -> bool:
        """检查两个张量的生命周期是否重叠"""
        return not (self.death_time <= other.birth_time or other.death_time <= self.birth_time)

class MemoryOptimizer:
    """基于图着色的内存优化器"""
    
    def __init__(self):
        self.tensors: List[TensorInfo] = []
        self.interference_graph = nx.Graph()
        self.coloring: Dict[str, int] = {}
        self.memory_pools: Dict[int, int] = {}  # 颜色 -> 内存池大小
        
    def add_tensor(self, name: str, size: int, birth_time: int, death_time: int):
        """添加张量信息"""
        tensor = TensorInfo(name, size, birth_time, death_time)
        self.tensors.append(tensor)
        
    def build_interference_graph(self):
        """构建干扰图（生命周期重叠的张量之间有边）"""
        self.interference_graph.clear()
        
        # 添加所有张量作为节点
        for tensor in self.tensors:
            self.interference_graph.add_node(tensor.name, size=tensor.size)
            
        # 为生命周期重叠的张量添加边
        for i, tensor1 in enumerate(self.tensors):
            for j, tensor2 in enumerate(self.tensors[i+1:], i+1):
                if tensor1.lifetime_overlaps(tensor2):
                    self.interference_graph.add_edge(tensor1.name, tensor2.name)
                    
    def greedy_coloring(self) -> Dict[str, int]:
        """贪心图着色算法"""
        # 按度数降序排列节点（度数高的节点优先着色）
        nodes_by_degree = sorted(self.interference_graph.nodes(), 
                               key=lambda x: self.interference_graph.degree(x), 
                               reverse=True)
        
        coloring = {}
        
        for node in nodes_by_degree:
            # 获取邻居节点已使用的颜色
            used_colors = set()
            for neighbor in self.interference_graph.neighbors(node):
                if neighbor in coloring:
                    used_colors.add(coloring[neighbor])
            
            # 找到最小的未使用颜色
            color = 0
            while color in used_colors:
                color += 1
            
            coloring[node] = color
            
        self.coloring = coloring
        return coloring
    
    def calculate_memory_pools(self) -> Dict[int, int]:
        """计算每个内存池所需的大小"""
        self.memory_pools = {}
        
        for tensor in self.tensors:
            color = self.coloring[tensor.name]
            if color not in self.memory_pools:
                self.memory_pools[color] = 0
            self.memory_pools[color] = max(self.memory_pools[color], tensor.size)
            
        return self.memory_pools
    
    def get_optimized_addresses(self, base_address: int = 0x10000000, 
                              alignment: int = 0x1000) -> Dict[str, int]:
        """
        生成优化后的地址映射
        
        参数:
        - base_address: 基础地址
        - alignment: 地址对齐字节数
        """
        addresses = {}
        current_address = base_address
        color_to_address = {}
        
        # 为每个颜色分配基地址
        for color in sorted(self.memory_pools.keys()):
            color_to_address[color] = current_address
            # 对齐到下一个边界
            pool_size = self.memory_pools[color]
            aligned_size = ((pool_size + alignment - 1) // alignment) * alignment
            current_address += aligned_size
            
        # 为每个张量分配地址
        for tensor in self.tensors:
            color = self.coloring[tensor.name]
            addresses[tensor.name] = color_to_address[color]
            
        return addresses
    
    def get_memory_savings(self, original_addresses: Dict[str, int]) -> Tuple[int, float]:
        """
        计算内存节省情况
        
        返回: (节省的字节数, 节省百分比)
        """
        # 计算原始内存使用（假设每个张量都有独立地址空间）
        original_memory = 0
        if original_addresses:
            max_addr = max(original_addresses.values())
            min_addr = min(original_addresses.values())
            original_memory = max_addr - min_addr
            
            # 加上最大张量的大小
            max_tensor_size = max(tensor.size for tensor in self.tensors)
            original_memory += max_tensor_size
        else:
            # 如果没有原始地址，假设每个张量占用独立空间
            original_memory = sum(tensor.size for tensor in self.tensors)
        
        # 计算优化后内存使用
        optimized_memory = sum(self.memory_pools.values())
        
        saved_memory = original_memory - optimized_memory
        save_percentage = (saved_memory / original_memory * 100) if original_memory > 0 else 0
        
        return saved_memory, save_percentage
    
    def visualize_interference_graph(self, save_path: str = None):
        """可视化干扰图"""
        plt.figure(figsize=(12, 8))
        
        # 节点颜色映射
        node_colors = [self.coloring.get(node, 0) for node in self.interference_graph.nodes()]
        
        # 绘制图
        pos = nx.spring_layout(self.interference_graph, k=1, iterations=50)
        nx.draw(self.interference_graph, pos, 
                node_color=node_colors, 
                with_labels=True, 
                node_size=500,
                font_size=8,
                cmap=plt.cm.Set3)
        
        plt.title("张量干扰图（相同颜色的节点可以共享内存）")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_optimization_report(self):
        """打印优化报告"""
        print("=" * 60)
        print("内存优化报告")
        print("=" * 60)
        
        print(f"张量总数: {len(self.tensors)}")
        print(f"内存池数量: {len(self.memory_pools)}")
        print(f"图的边数: {self.interference_graph.number_of_edges()}")
        
        print("\n内存池分配:")
        total_optimized = 0
        for color, size in sorted(self.memory_pools.items()):
            print(f"  池 {color}: {size:,} 字节 ({size/1024/1024:.2f} MB)")
            total_optimized += size
            
        print(f"\n优化后总内存: {total_optimized:,} 字节 ({total_optimized/1024/1024:.2f} MB)")
        
        # 计算节省情况
        original_total = sum(tensor.size for tensor in self.tensors)
        saved = original_total - total_optimized
        save_percentage = (saved / original_total * 100) if original_total > 0 else 0
        
        print(f"节省内存: {saved:,} 字节 ({saved/1024/1024:.2f} MB)")
        print(f"节省比例: {save_percentage:.1f}%")


def parse_yolo_inference_code(code_content: str) -> List[Dict]:
    """
    解析YOLOv10推理代码，提取张量信息
    这是一个简化的解析器，实际使用时需要根据具体代码格式调整
    """
    operations = []
    operation_count = 0
    
    # 简化的模式匹配（实际实现需要更复杂的解析）
    lines = code_content.split('\n')
    
    for i, line in enumerate(lines):
        if 'QLinearConv_AUTO' in line or 'QLinearConcat' in line or 'QLinearAdd' in line:
            # 提取地址信息（这里需要根据实际代码格式调整）
            if '0x' in line:
                addresses = []
                parts = line.split(',')
                for part in parts:
                    if '0x' in part:
                        addr_str = part.strip()
                        if addr_str.startswith('0x'):
                            addresses.append(addr_str)
                
                if len(addresses) >= 2:  # 至少有输入和输出地址
                    operations.append({
                        'operation': operation_count,
                        'input_addr': addresses[0] if len(addresses) > 0 else None,
                        'output_addr': addresses[1] if len(addresses) > 1 else None,
                        'line_number': i + 1
                    })
                    operation_count += 1
    
    return operations


# 数学公式和算法说明
ALGORITHM_DESCRIPTION = """
基于图着色的内存优化算法数学模型

1. 问题建模:
   设 T = {t₁, t₂, ..., tₙ} 为所有张量的集合
   每个张量 tᵢ 有以下属性:
   - sᵢ: 张量大小（字节）
   - bᵢ: 生命周期开始时间
   - dᵢ: 生命周期结束时间

2. 干扰图构建:
   构建无向图 G = (V, E)，其中:
   - V = T（张量集合作为顶点）
   - E = {(tᵢ, tⱼ) | 生命周期重叠(tᵢ, tⱼ)}
   
   生命周期重叠条件:
   overlap(tᵢ, tⱼ) = ¬(dᵢ ≤ bⱼ ∨ dⱼ ≤ bᵢ)

3. 图着色问题:
   寻找映射 c: V → ℕ，使得:
   ∀(tᵢ, tⱼ) ∈ E, c(tᵢ) ≠ c(tⱼ)
   
   目标: 最小化颜色数量 |{c(tᵢ) | tᵢ ∈ V}|

4. 内存池大小计算:
   对于颜色 k，内存池大小为:
   poolₖ = max{sᵢ | c(tᵢ) = k}

5. 总内存使用:
   M_optimized = Σₖ poolₖ

6. 内存节省率:
   假设原始内存使用为 M_original = Σᵢ sᵢ
   节省率 = (M_original - M_optimized) / M_original × 100%

7. 贪心着色算法复杂度:
   时间复杂度: O(|V|² + |V||E|) = O(n² + nm)
   空间复杂度: O(|V| + |E|) = O(n + m)
   其中 n = |V|, m = |E|

8. 地址分配公式:
   对于颜色 k，基地址为:
   base_addrₖ = base_addr + Σ⁰ᵏ⁻¹ align(poolⱼ)
   
   其中 align(x) = ⌈x/alignment⌉ × alignment
"""

if __name__ == "__main__":
    # 示例使用
    optimizer = MemoryOptimizer()
    
    # 添加示例张量（模拟YOLOv10中的一些张量）
    # 格式: (名称, 大小(MB), 生命开始, 生命结束)
    example_tensors = [
        ("node1", 1024*1024, 0, 2),      # 1MB, 时间0-2
        ("node2", 2048*1024, 1, 5),      # 2MB, 时间1-5  
        ("node3", 1024*1024, 3, 6),      # 1MB, 时间3-6
        ("node4", 3072*1024, 4, 8),      # 3MB, 时间4-8
        ("node5", 1536*1024, 6, 10),     # 1.5MB, 时间6-10
        ("node6", 2048*1024, 7, 12),     # 2MB, 时间7-12
        ("concat1", 4096*1024, 9, 13),   # 4MB, 时间9-13
        ("output", 1024*1024, 11, 15),   # 1MB, 时间11-15
    ]
    
    for name, size, birth, death in example_tensors:
        optimizer.add_tensor(name, size, birth, death)
    
    # 执行优化
    optimizer.build_interference_graph()
    coloring = optimizer.greedy_coloring()
    memory_pools = optimizer.calculate_memory_pools()
    addresses = optimizer.get_optimized_addresses()
    
    # 打印结果
    optimizer.print_optimization_report()
    
    print("\n优化后的地址分配:")
    for tensor_name, address in addresses.items():
        print(f"  {tensor_name}: 0x{address:08x}")
        
    print("\n" + "="*60)
    print("算法说明:")
    print(ALGORITHM_DESCRIPTION)
