#define DTYPE float // 取决于模型，yolov5_gray_640.onnx使用的是32位浮点数
#pragma once

struct Tensor
{
    int ndim;
    int *lens;   // 每一维的长度
    DTYPE *data; // 多维数据
};

struct TensorQ
{
    int ndim;
    int *lens; // 每一维的长度
    int *data; // 多维数据,使用定点数表示实数
};
