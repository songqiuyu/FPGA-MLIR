graph TD
    Start((开始)) --> Init[初始化基础常量参数\nBASE_ADDR, BASE_OFFSET, WEIGHT_BASE 等]
    Init --> LoadFile[读取输入 .mlir 文件\n加载模型权重 .npz 文件]

    %% 预处理阶段
    subgraph 预处理阶段: 权重地址预分配与分块参数计算
        LoadFile --> RegexConv[正则表达式匹配所有 "coa.qlinearconv" 操作]
        RegexConv --> LoopConv{遍历提取到的Conv操作}
        
        LoopConv -- "存在未处理项" --> CalcWeightSz[计算权重与偏置大小\nInput Channels 向上对齐到 32]
        CalcWeightSz --> AssignWeightAddr[递增分配 weight_addr 与 bias_addr\n存入 weight_info 字典]
        AssignWeightAddr --> ExtractParams[从 MLIR 文本提取 stride, pad, dilation, kernel]
        ExtractParams --> CallGetTile[调用 get_tile() 获取硬件最优的分块大小 tM, tR, tC]
        CallGetTile --> SaveTile[将分块结果存入 tile_info 字典]
        SaveTile --> LoopConv
        
        LoopConv -- "遍历结束" --> CheckFC{检查 .npz 是否包含 FC 层?}
        CheckFC -- "是" --> ProcessFC[计算并分配全连接层的 weight/bias 地址\nInput Dimension 向上对齐到 32]
        ProcessFC --> MainLoop
        CheckFC -- "否" --> MainLoop
    end

    %% 辅助逻辑子图
    subgraph 辅助计算: get_tile 内存限额判定逻辑
        TileStart((输入维度参数)) -.-> CheckBuffer[计算 wdepth, gdepth, odepth]
        CheckBuffer -.-> OverLimit{判断 Buffer 状态标识}
        OverLimit -- "Flag 1" -.-> ReduceTM[wdepth 超限\n逐步缩减 tM]
        OverLimit -- "Flag 2 或 3" -.-> ReduceTRC[gdepth 或 odepth 超限\n优先缩减 tR, 其次缩减 tC]
        OverLimit -- "Flag 0" -.-> TryIncrease[未超限\n尝试按倍数扩大 tR 寻找最优解]
        ReduceTM -.-> CheckBuffer
        ReduceTRC -.-> CheckBuffer
        TryIncrease -.-> TileEnd((返回 tM, tR, tC))
    end
    
    CallGetTile -.-> TileStart
    TileEnd -.-> CallGetTile

    %% 核心处理阶段
    subgraph 核心处理: MLIR 逐行解析与地址属性重写
        MainLoop{按行遍历 MLIR 文本}
        MainLoop -- "已读完" --> WriteOutput[将修改后的全部行写入输出 .mlir 文件]
        
        MainLoop -- "读取新行" --> LineTypeCheck{检查操作类型 (op_type)}
        LineTypeCheck -- "空行 / 非目标操作" --> Skip[保留原样, 加入 new_lines]
        Skip --> MainLoop
        
        LineTypeCheck -- "qlinearconv / maxpool\nqlinearadd / qgemm 等" --> ParseIO[解析输出变量与所有的输入变量]
        ParseIO --> CalcTensorsize[根据 shape_estimate 估算 I/O Tensor 大小]
        CalcTensorsize --> AssignBaseAddr[计算基础 in_addr 与 out_addr]
        AssignBaseAddr --> OffsetCheck{判定是否为网络第一层 (first_op)}
        
        OffsetCheck -- "是" --> AddrFirstOp[in_addr = 0x0\nout_addr = Base + 0x10000000]
        OffsetCheck -- "否" --> AddrNormalOp[in_addr = Base + 0x10000000\nout_addr = Base + 0x10000000]
        
        AddrFirstOp --> OpSwitch
        AddrNormalOp --> OpSwitch
        
        OpSwitch{根据具体 Op_Type 提取并计算附属属性}
        
        OpSwitch -- "qlinearadd" --> GenAddAttr[获取两个输入的地址\n提取 scale 并计算 factor, factor2 量化乘子]
        OpSwitch -- "qlinearconv / qgemm" --> GenConvAttr[查表读取 weight/bias/silu 地址及大小\n张量维度补齐: M对齐16, N对齐32\n载入之前计算的 tM, tR, tC\n计算 factor 量化乘子]
        OpSwitch -- "其他 (如 maxpool)" --> GenNormalAttr[仅构建标准的输入输出地址和大小信息]
        
        GenAddAttr --> InjectAttr
        GenConvAttr --> InjectAttr
        GenNormalAttr --> InjectAttr
        
        InjectAttr[格式化为字符串 addr_attrs\n拼接并注入当前 MLIR 行的末尾 `{}` 中]
        InjectAttr --> SetFirstOpFlag[置 first_op = False]
        SetFirstOpFlag --> MainLoop
    end

    WriteOutput --> End((程序结束))

    %% 样式设定
    style Start fill:#f0f4c3,stroke:#827717,stroke-width:2px
    style End fill:#f0f4c3,stroke:#827717,stroke-width:2px
    style LoopConv fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style MainLoop fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style OffsetCheck fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style OpSwitch fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style OverLimit fill:#ffebee,stroke:#c62828,stroke-width:2px