#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "basic.h"
#include <stdio.h>
#include "tensor.h"
#include <stdint.h>

// long long getfactor(double f, int pow_n)
// {
//     long long m = 0;
//     unsigned char *t = (unsigned char *)&f;
//     // get e value
//     int e = 0;
//     e |= (t[7] & 0x7f) << 4;
//     e |= (t[6]) >> 4;
//     e -= 1023;
//     // get m value
//     m |= 0x10000000000000;
//     m |= t[0];
//     m |= t[1] << 8;
//     m |= (long long)(t[2]) << 16;
//     m |= (long long)(t[3]) << 24;
//     m |= (long long)(t[4]) << 32;
//     m |= (long long)(t[5]) << 40;
//     m |= (long long)(t[6] & 0x0f) << 48;

//     // printf("before shift=%lld\n",m);
//     //  int yhas1=0;
//     int shift = pow_n + e - 52;
//     int xhas1 = 0;
//     // printf("shift=%d\n",shift);
//     if (shift < 0)
//     {
//         for (int i = 0; i < (-shift) - 1; i++)
//         {
//             m = m >> 1;
//         }
//         if (m & 1)
//             xhas1 = 1;
//         m = m >> 1;
//     }
//     else if (shift > 0)
//     {
//         for (int i = 0; i < shift; i++)
//         {
//             m = m << 1;
//         }
//     }

//     if (f < 0)
//         m = -m;
//     if (xhas1)
//         m += 1;
//     // printf("factor=%lld\n", m);
//     return m;
// }

long long getfactor(double x, int bits)
{
    // printf("getfactor function: %lld\n", (long long)(x * pow(2.0, bits)));
    long long factor = (long long)(x * pow(2.0, bits));
    if(factor > 2147483647 || factor < -2147483648){
        printf("error factor!%lld\n", factor);
    }
    return factor;
}

int getpown(double f)
{
    return -28;
    /* long long ret = 0;
    unsigned char *t = (unsigned char *)&f;
    ret |= 0x10000000000000;
    ret |= t[0];
    ret |= t[1] << 8;
    ret |= (long long)(t[2]) << 16;
    ret |= (long long)(t[3]) << 24;
    ret |= (long long)(t[4]) << 32;
    ret |= (long long)(t[5]) << 40;
    ret |= (long long)(t[6] & 0x0f) << 48;

    int iadd = 0;
    for (int i = 0; i < 31; i++)
    {
        ret = ret >> 1;
        iadd++;
    }

    int pown = 0;
    pown |= (t[7] & 0x7f) << 4;
    pown |= (t[6]) >> 4;
    pown -= 1023;
    printf("pown=%d\n", pown - 52 + iadd);
    return pown - 52 + iadd; */
}

int factor_product(int x, long long factor, int pown)
{
    long long t = x * factor;
    // int yhas1 = 0;
    for (int i = 0; i < pown - 1; i++)
    {
        // yhas1 |= t & 1;
        t = t >> 1;
    }
    int xhas1 = t & 1;
    t = t >> 1;
    int carry = xhas1; //&& yhas1;
    return t + carry;
}

// more precise
int factor_product_add(int x, long long factor, int pown, int y, long long factor2, int pown2)
{
    long long t = x * factor;
    long long t2 = y * factor2;
    int commonpown = 0;
    int xhas1 = 0;
    int diff = 0;
    if (pown2 > pown)
    {
        commonpown = pown;
        diff = pown2 - pown;
        for (int i = 0; i < diff - 1; i++)
        {
            // yhas1 |= t & 1;
            t2 = t2 >> 1;
        }
        xhas1 = t2 & 1;
        t2 = t2 >> 1;
        if (xhas1)
            t2 = t2 + 1;
    }
    else if (pown > pown2)
    {
        commonpown = pown2;
        diff = pown - pown2;
        for (int i = 0; i < diff - 1; i++)
        {
            // yhas1 |= t & 1;
            t = t >> 1;
        }
        xhas1 = t & 1;
        t = t >> 1;
        if (xhas1)
            t = t + 1;
    }
    else
    {
        commonpown = pown2;
    }
    // int yhas1 = 0;
    long long add = t + t2;
    for (int i = 0; i < commonpown - 1; i++)
    {
        // yhas1 |= t & 1;
        add = add >> 1;
    }
    xhas1 = add & 1;
    add = add >> 1;
    int carry = xhas1; //&& yhas1;
    return add + carry;
}

int result_product_gen_32bit(int x, long long factor, int zo)
{
    // long long data = x * factor;
    // uint32_t sign_data_t2 = (data >= 0) ? 0 : 1;
    // data = data < 0 ? -data : data;
    // uint32_t integer_data_t2 = data >> 32;
    // uint32_t fractional_data_t2 = data & 0xFFFFFFFF;

    // int result_data_old_v2;
    // // 四舍五入
    // if (fractional_data_t2 >= 2147483648)
    // {
    //     result_data_old_v2 = integer_data_t2 + 1;
    // }
    // else
    // {
    //     result_data_old_v2 = integer_data_t2;
    // }

    // // 如果是负数，取反
    // if (sign_data_t2)
    // {
    //     result_data_old_v2 = -result_data_old_v2;
    // }
    // result_data_old_v2 = result_data_old_v2 + zo;
    // return result_data_old_v2;

    long long data_t_float = x * factor;

    float data_t_f = (float)data_t_float / (float)(68719476736);
    // float data_t_f = (float)data_t_float / (float)(1099511627776);
    int result_uint2 = nearbyint(data_t_f) + zo;

    long long data_t = x * factor;
    uint32_t sign_data_t = (data_t >= 0) ? 0 : 1;
    data_t = data_t < 0 ? -data_t : data_t;
    uint64_t integer_data_t = data_t >> 36;
    uint64_t fractional_data_t = data_t & 0xFFFFFFFFF;

    uint32_t data_ieee754;
    uint64_t fractional_data_t2 = (fractional_data_t >> 13);
    convertToIEEE754(sign_data_t, integer_data_t, fractional_data_t2, &data_ieee754);
    int result = my_nearbyint(data_ieee754) + zo;
    return result;
}
int result_product_add_gen_32bit(int x, long long factor, int y, long long factor2, int zx, int zw, int zo)
{
    //---------------version1------------------------------
    // long long data_t1 = (x - zx) * factor;
    // long long data_t2 = (y - zw) * factor2;
    // long long data_t = data_t1 + data_t2;
    // uint32_t sign_data_t = (data_t >= 0) ? 0 : 1;
    // data_t = data_t < 0 ? -data_t : data_t;
    // uint32_t integer_data_t = data_t >> 32;
    // uint32_t fractional_data_t = data_t & 0xFFFFFFFF;
    // int result_data;
    // if (fractional_data_t >= 2147483648)
    // {
    //     result_data = integer_data_t + 1;
    // }
    // else
    // {
    //     result_data = integer_data_t;
    // }

    // if (sign_data_t)
    // {
    //     result_data = -result_data;
    // }

    // result_data = result_data + zo + zo - zo;
    // return result_data;

    // long long data_t1 = (x - zx) * factor;
    // long long data_t2 = (y - zw) * factor2;
    // long long data_result = data_t1 + data_t2;

    // float data_t_f = (float)data_result / (float)(68719476736);
    // // float data_t_f = (float)data_t_float / (float)(68719476736);
    // int result_uint2 = nearbyint(data_t_f) + zo;

    // uint32_t sign_result = (data_result >= 0) ? 0 : 1;
    // data_result = data_result < 0 ? -data_result : data_result;
    // uint64_t integer_data_result = data_result >> 36;
    // uint64_t fractional_data_result = data_result & 0xFFFFFFFFF;

    // uint32_t data_ieee754;
    // uint64_t fraction_data_result_2 = (fractional_data_result >> 13);
    // convertToIEEE754(sign_result, integer_data_result, fraction_data_result_2, &data_ieee754);
    // int result = my_nearbyint(data_ieee754) + zo + zo - zo;
    // return result;

    // IEEE add zo
    long long data_t1 = (x - zx) * factor;
    long long data_t2 = (y - zw) * factor2;
    long long data_result = data_t1 + data_t2;

    float data_t_f = (float)data_result / (float)(268435456);
    int result_uint_t_f = nearbyint(data_t_f) + zo;
    int result_uint_t_f2 = nearbyint(data_t_f + zo);
    if(result_uint_t_f != result_uint_t_f2){
        // printf("here!\n");
    }

    uint32_t sign_result = (data_result >= 0) ? 0 : 1;
    data_result = data_result < 0 ? -data_result : data_result;
    uint64_t integer_data_result = data_result >> 28;
    uint64_t fractional_data_result = data_result & 0xFFFFFFF;
    uint32_t data_ieee754;
    uint64_t fraction_data_result_2 = (fractional_data_result >> 5);
    if(integer_data_result > 255){
        // printf("here\n");
    }
    convertToIEEE754(sign_result, integer_data_result, fraction_data_result_2, &data_ieee754);
    uint32_t data_sign = sign_result;
    uint32_t data_exponent = (data_ieee754 >> 23) & 0xFF;
    uint32_t data_mantissa = data_ieee754 & 0x7FFFFF;

    uint32_t zo_ieee754;
    uint32_t zo_sign = (zo >= 0) ? 0 : 1;
    uint32_t zo_new = zo < 0 ? -zo : zo;
    convertToIEEE754(zo_sign, zo_new, 0, &zo_ieee754);
    uint32_t zo_exponent = (zo_ieee754 >> 23) & 0xFF;
    uint32_t zo_mantissa = zo_ieee754 & 0x7FFFFF;


    // data_sign = 0;
    // data_exponent = 131;
    // data_mantissa = 3407877;

    // zo_sign = 1;
    // zo_exponent = 134;
    // zo_mantissa = 0;

    // data_sign = 0;
    // data_exponent = 131;
    // data_mantissa = 2359300;

    // zo_sign = 1;
    // zo_exponent = 134;
    // zo_mantissa = 0;
    
    
    uint32_t result_add = ieee754_float_add(data_sign, data_exponent, data_mantissa, zo_sign, zo_exponent, zo_mantissa);
    int result = my_nearbyint(result_add);

    if (result != result_uint_t_f2) {
        // printf("error!\n");
    }
    return result;
}

int my_test()
{
    int zo = -128;
    uint32_t zo_ieee754;
    uint32_t zo_sign = (zo >= 0) ? 0 : 1;
    uint32_t zo_new = zo < 0 ? -zo : zo;
    convertToIEEE754(zo_sign, zo_new, 0, &zo_ieee754);
    uint32_t zo_exponent = (zo_ieee754 >> 23) & 0xFF;
    uint32_t zo_mantissa = zo_ieee754 & 0x7FFFFF;

    // long long data_result = 2783139131333;
    // int zo = -128;
    // float data_t_f = (float)data_result / (float)(68719476736);
    // // float data_t_f = (float)data_t_float / (float)(68719476736);

    // int result_uint2 = nearbyint(data_t_f) + zo;
    // int result_uint22 = nearbyint(data_t_f + (float)zo);

    // uint32_t sign_zo = (zo >= 0) ? 0 : 1;
    // zo = zo < 0 ? -zo : zo;
    // uint32_t zo_ieee754;
    // convertToIEEE754(sign_zo, zo_ieee754, 0, &zo_ieee754);

    // uint32_t sign_result = (data_result >= 0) ? 0 : 1;
    // data_result = data_result < 0 ? -data_result : data_result;
    // uint64_t integer_data_result = data_result >> 36;
    // uint64_t fractional_data_result = data_result & 0xFFFFFFFFF;

    // uint32_t data_ieee754;
    // uint64_t fraction_data_result_2 = (fractional_data_result >> 13);
    // convertToIEEE754(sign_result, integer_data_result, fraction_data_result_2, &data_ieee754);
    // int result = my_nearbyint(data_ieee754) + zo + zo - zo;

    // uint32_t data_sign_verilog = sign_result;
    // uint32_t data_exponent_bits_verilog = (data_ieee754 >> 23) & 0xFF; // 指数部分
    // uint32_t data_mantissa_bits_verilog = data_ieee754 & 0x7FFFFF;     // 尾数部分

    // uint32_t zo_sign_verilog = (zo_ieee754 >> 31) & 0x1;
    // uint32_t zo_exponent_bits_verilog = (zo_ieee754 >> 23) & 0xFF; // 指数部分
    // uint32_t zo_mantissa_bits_verilog = zo_ieee754 & 0x7FFFFF;

    // uint32_t result_add = ieee754_float_add(data_sign_verilog, data_exponent_bits_verilog, data_mantissa_bits_verilog, zo_sign_verilog, zo_exponent_bits_verilog, zo_mantissa_bits_verilog);
    // int result2 = my_nearbyint(result_add);

    // return result;
}

#include <stdio.h>
#include <stdint.h>

// 假设 mantissa1, mantissa2 是 24 位尾数（含隐含的 1），exp1, exp2 是指数
void align_exponents(uint32_t *mantissa1, int *exp1, uint32_t *mantissa2, int *exp2) {
    int exp_diff = *exp1 - *exp2; // 指数差

    if (exp_diff > 0) {
        // 对 mantissa2 右移
        int shift = exp_diff;
        if (shift > 0) {
            // 提取将被移出的位
            uint32_t lost_bits = *mantissa2 & ((1 << shift) - 1);
            // 右移
            *mantissa2 >>= shift;
            // 舍入判断
            uint32_t round_bit = (lost_bits >> (shift - 1)) & 1; // 移出部分的最高位
            // uint32_t sticky_bits = lost_bits & ((1 << (shift - 1)) - 1); // 剩余低位
            // if (round_bit && (sticky_bits || (*mantissa2 & 1))) { // 进位条件
            if (round_bit  || (*mantissa1 & 1)) { // 进位条件
                (*mantissa1)++;
            }
        }
        *exp2 += exp_diff; // 调整指数
    } else if (exp_diff < 0) {
        // 对 mantissa1 右移
        int shift = -exp_diff; // 取绝对值
        if (shift > 0) {
            // 提取将被移出的位
            uint32_t lost_bits = *mantissa1 & ((1 << shift) - 1);
            // 右移
            *mantissa1 >>= shift;
            // 舍入判断
            uint32_t round_bit = (lost_bits >> (shift - 1)) & 1; // 移出部分的最高位
            // uint32_t sticky_bits = lost_bits & ((1 << (shift - 1)) - 1); // 剩余低位
            // if (round_bit && (sticky_bits || (*mantissa1 & 1))) { // 进位条件
            //     (*mantissa1)++;
            // }
            if (round_bit  || (*mantissa1 & 1)) { // 进位条件
                (*mantissa1)++;
            }
        }
        *exp1 += -exp_diff; // 调整指数
    }
}

// // 用于调试的二进制打印函数
// void print_binary_v2(uint32_t num, int bits) {
//     for (int i = bits - 1; i >= 0; i--) {
//         printf("%d", (num >> i) & 1);
//         if (i % 4 == 0) printf(" ");
//     }
//     printf("\n");
// }


uint32_t ieee754_float_add(uint32_t sign1, uint32_t exponent1, uint32_t significant1,
                           uint32_t sign2, uint32_t exponent2, uint32_t significant2)
{
    // 处理特殊情况
    if (exponent1 == 0xFF || exponent2 == 0xFF)
    {
        // 如果其中一个数是NaN或无穷大
        if (exponent1 == 0xFF && significant1 != 0)
            return (sign1 << 31) | 0x7FC00000; // NaN
        if (exponent2 == 0xFF && significant2 != 0)
            return (sign2 << 31) | 0x7FC00000; // NaN
        if (exponent1 == 0xFF && exponent2 == 0xFF && sign1 != sign2)
            return 0x7FC00000;                                                                    // 无穷大相减，返回NaN
        return (exponent1 == 0xFF) ? ((sign1 << 31) | 0x7F800000) : ((sign2 << 31) | 0x7F800000); // 返回无穷大
    }

    // 如果其中一个数为0，直接返回另一个数
    if (exponent1 == 0 && significant1 == 0)
        return (sign2 << 31) | (exponent2 << 23) | significant2;
    if (exponent2 == 0 && significant2 == 0)
        return (sign1 << 31) | (exponent1 << 23) | significant1;

    // 添加隐含的1到尾数
    uint32_t mantissa1 = (exponent1 == 0) ? significant1 : (significant1 | 0x00800000);
    uint32_t mantissa2 = (exponent2 == 0) ? significant2 : (significant2 | 0x00800000);

    uint32_t mantissa_test;

    // 对齐指数
    int32_t exp1 = (exponent1 == 0) ? -126 : (exponent1 - 127); // 实际指数
    int32_t exp2 = (exponent2 == 0) ? -126 : (exponent2 - 127);
    int32_t exp_diff = exp1 - exp2;

    // if (exp_diff > 0)
    // {
    //     mantissa2 >>= exp_diff; // 右移较小的尾数
    //     exp2 += exp_diff;
    // }
    // else if (exp_diff < 0)
    // {
    //     // mantissa1 >>= -exp_diff; // 右移较小的尾数

    //     mantissa1 >>= (-exp_diff); // 右移较小的尾数
    //     // mantissa_test = mantissa1 >> -exp_diff;
    //     // mantissa1 = significant1 >> -exp_diff;
    //     exp1 += -exp_diff;
    // }

    align_exponents(&mantissa1, &exp1, &mantissa2, &exp2);

    // 尾数相加
    uint32_t result_mantissa;
    uint32_t result_sign;
    int32_t result_exponent = exp1;

    if (sign1 == sign2)
    {
        result_mantissa = mantissa1 + mantissa2; // 同号相加
        result_sign = sign1;
    }
    else
    {
        if (mantissa1 > mantissa2)
        {
            result_mantissa = mantissa1 - mantissa2; // 异号相减
            result_sign = sign1;
        }
        else
        {
            result_mantissa = mantissa2 - mantissa1; // 异号相减
            result_sign = sign2;
        }
    }

    // 规格化结果
    if (result_mantissa & 0x01000000)
    { // 如果尾数溢出
        result_mantissa >>= 1;
        result_exponent++;
    }
    while ((result_mantissa & 0x00800000) == 0 && result_exponent > -126)
    { // 如果尾数不足
        result_mantissa <<= 1;
        result_exponent--;
    }

    // 检查溢出或下溢
    if (result_exponent >= 0xFF)
    {                                            // 溢出
        return (result_sign << 31) | 0x7F800000; // 返回无穷大
    }
    if (result_exponent < -126)
    {                               // 下溢
        return (result_sign << 31); // 返回0
    }

    // 组合结果
    uint32_t result_exponent_biased = (result_exponent + 127) << 23;
    uint32_t result_significant = result_mantissa & 0x007FFFFF; // 去掉隐含的1
    return (result_sign << 31) | result_exponent_biased | result_significant;
}

int getindex(int ndims, ...)
{
    int index = 0;
    va_list ap;
    va_start(ap, ndims);
    int *xi = (int *)malloc(sizeof(int) * ndims);
    for (int i = 0; i < ndims; i++)
    {
        xi[i] = va_arg(ap, int);
    }
    int *dims = va_arg(ap, int *);
    for (int i = 0; i < ndims; i++)
    {
        int product = xi[i];
        for (int k = i + 1; k < ndims; k++)
        {
            product *= dims[k];
        }
        index += product;
    }
    free(xi);
    return index;
}

void convert_index(int index, int batch_size, int rows, int cols, int channels) {
    // 首先，我们需要计算在新的顺序下的步长
    int stride_c = rows * cols;
    int stride_r = cols;
    int total_size = batch_size * channels * rows * cols;
    
    // 从原始index中提取坐标
    int temp_index = index;
    int out_batch,out_channels,out_rows,out_cols;
    
    // 计算batch
    out_batch = temp_index / (channels * rows * cols);
    temp_index %= (channels * rows * cols);
    
    // 计算channels
    out_channels = temp_index / (rows * cols);
    temp_index %= (rows * cols);
    
    // 计算rows
    out_rows = temp_index / cols;
    
    // 计算columns
    out_cols = temp_index % cols;
    printf("convert_index: <%d,%d,%d,%d>\n", out_batch, out_channels, out_rows, out_cols);
}

struct Tensor *getTensor(int ndim)
{
    struct Tensor *ret = (struct Tensor *)malloc(sizeof(struct Tensor));
    ret->ndim = ndim;
    ret->lens = (int *)malloc(sizeof(int) * ret->ndim);
    return ret;
}

struct TensorQ *getTensorQ(int ndim)
{
    struct TensorQ *ret = (struct TensorQ *)malloc(sizeof(struct TensorQ));
    ret->ndim = ndim;
    ret->lens = (int *)malloc(sizeof(int) * ret->ndim);
    return ret;
}

void freeTensor(struct Tensor *tensor)
{
    free(tensor->data);
    free(tensor->lens);
    free(tensor);
}

void freeTensorQ(struct TensorQ *tensor)
{
    free(tensor->data);
    free(tensor->lens);
    free(tensor);
}

void printshape(struct Tensor *t)
{
    printf("<");
    for (int i = 0; i < t->ndim; i++)
    {
        if (i != t->ndim - 1)
            printf("%d,", t->lens[i]);
        else
            printf("%d", t->lens[i]);
    }
    printf(">\n");
}

void printshapeQ(struct TensorQ *t)
{
    printf("<");
    for (int i = 0; i < t->ndim; i++)
    {
        if (i != t->ndim - 1)
            printf("%d,", t->lens[i]);
        else
            printf("%d", t->lens[i]);
    }
    printf(">\n");
}

unsigned char *readfile(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    int flen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = (unsigned char *)malloc(flen);
    fread(data, 1, flen, fp);
    fclose(fp);
    return data;
}

int filesize(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    int flen = ftell(fp);
    return flen;
}

int getlength(struct Tensor *x)
{
    int length = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        length *= x->lens[i];
    }
    return length;
}

int getlengthQ(struct TensorQ *x)
{
    int length = 1;
    for (int i = 0; i < x->ndim; i++)
    {
        length *= x->lens[i];
    }
    return length;
}

int getaxilen(struct Tensor *x, int axi) // total length of an axi,also considering length of itself
{
    if (axi >= x->ndim)
        return 1;
    int len = 1;
    for (int i = axi; i < x->ndim; i++)
    {
        len *= x->lens[i];
    }
    return len;
}

int getaxilenQ(struct TensorQ *x, int axi) // total length of an axi,also considering length of itself
{
    if (axi >= x->ndim)
        return 1;
    int len = 1;
    for (int i = axi; i < x->ndim; i++)
    {
        len *= x->lens[i];
    }
    return len;
}

void compareTensor(struct Tensor *x, struct Tensor *y)
{
    if (x->ndim != y->ndim)
    {
        printf("dim not equal\n");
        return;
    }
    for (int i = 0; i < x->ndim; i++)
    {
        if (x->lens[i] != y->lens[i])
        {
            printf("shape not equal\n");
            return;
        }
    }
    int len = getlength(x);
    for (int i = 0; i < len; i++)
    {
        if (x->data[i] != y->data[i])
        {
            printf("value not equal %f!=%f at %d\n", x->data[i], y->data[i], i);
            return;
        }
    }
    printf("equal\n");
}

struct Tensor *copyTensor(struct Tensor *x)
{
    struct Tensor *ret = (struct Tensor *)malloc(sizeof(struct Tensor));
    ret->ndim = x->ndim;
    ret->lens = (int *)malloc(sizeof(int) * ret->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int len = getlength(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * len);
    memcpy(ret->data, x->data, len * sizeof(DTYPE));
    return ret;
}

struct TensorQ *copyTensorQ(struct TensorQ *x)
{
    struct TensorQ *ret = (struct TensorQ *)malloc(sizeof(struct TensorQ));
    ret->ndim = x->ndim;
    ret->lens = (int *)malloc(sizeof(int) * ret->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int len = getlengthQ(x);
    ret->data = (int *)malloc(sizeof(int) * len);
    memcpy(ret->data, x->data, len * sizeof(int));
    return ret;
}

struct Tensor *convertTensor(struct TensorQ *x, double s, int z)
{
    struct Tensor *ret = (struct Tensor *)malloc(sizeof(struct Tensor));
    ret->ndim = x->ndim;
    ret->lens = (int *)malloc(sizeof(int) * ret->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int len = getlengthQ(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * len);
    for (int i = 0; i < len; i++)
    {
        ret->data[i] = s * (x->data[i] - z);
    }
    return ret;
}

void cliptensorQ(struct TensorQ *x)
{
    int len = getlengthQ(x);
    for (int i = 0; i < len; i++)
    {
        if (x->data[i] > 127)
            x->data[i] = 127;
        if (x->data[i] < -128)
            x->data[i] = -128;
    }
}

int calculate_buffer_consumption(int tN, int tM, int tR, int tC, int N, int M, int kernel, int stride, int pad, int dilation)
{
    // printf("%d,%d,%d,%d\n", tN, tM, tR, tC);
    int tN32_rd = 0; // rounded
    for (int sn = 0; sn < N; sn += tN)
    {
        int t = ((sn + tN + 31) >> 5) - (sn >> 5);
        if (t > tN32_rd)
            tN32_rd = t;
    }
    int tM32_rd = 0;
    for (int sm = 0; sm < M; sm += tM)
    {
        int t = ((sm + tM + 15) >> 4) - (sm >> 4);
        if (t > tM32_rd)
            tM32_rd = t;
    }
    int relems = (tR - 1) * stride + (kernel - 1) * dilation + 1;
    int celems = (tC - 1) * stride + (kernel - 1) * dilation + 1;
    int gdepth = relems * celems * tN32_rd;
    int wbuf_single_m_size = tN32_rd * kernel * kernel;
    int wdepth = wbuf_single_m_size * tM32_rd;
    int odepth = tR * tC * tM32_rd;
    // printf("%d,%d,%d\n", gdepth, wdepth, odepth);
    // system("pause");
    if (wdepth >= 256)
    {
        return 1;
    }
    if (gdepth >= 1024)
    {
        return 2;
    }
    if (odepth >= 2048)
    {
        return 3;
    }
    return 0;
}

int getslice(struct TensorQ *x, struct TensorQ *w, int pad, int d, int s, int tX[3])
{
    int R = (x->lens[2] + pad * 2 - (d * (w->lens[2] - 1) + 1)) / s + 1;
    int C = (x->lens[3] + pad * 2 - (d * (w->lens[3] - 1) + 1)) / s + 1;
    int M = w->lens[0];
    int N = w->lens[1];
    int tM = M;
    int tR = R;
    int tC = C;

    int flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
    while (flag != 0)
    {
        if (flag == 1)
        {
            if (tM % 2 == 0)
            {
                tM = tM / 2;
            }
            else if (tM % 3 == 0)
            {
                tM = tM / 3;
            }
            else if (tM % 4 == 0)
            {
                tM = tM / 4;
            }
            else if (tM % 5 == 0)
            {
                tM = tM / 5;
            }
            else
            {
                tM = 16;
            }
            if (tM < 16)
                tM = 16;
        }
        if (flag == 2)
        {
            if (tR != 1)
            {
                if (tR % 2 == 0)
                {
                    tR = tR / 2;
                }
                else if (tR % 3 == 0)
                {
                    tR = tR / 3;
                }
                else if (tR % 4 == 0)
                {
                    tR = tR / 4;
                }
                else if (tR % 5 == 0)
                {
                    tR = tR / 5;
                }
                else if (tR % 6 == 0)
                {
                    tR = tR / 6;
                }
                else
                {
                    tR = 1;
                }
            }
            else
            {
                if (tC % 2 == 0)
                {
                    tC = tC / 2;
                }
                else if (tC % 3 == 0)
                {
                    tC = tC / 3;
                }
                else if (tC % 4 == 0)
                {
                    tC = tC / 4;
                }
                else if (tC % 5 == 0)
                {
                    tC = tC / 5;
                }
                else if (tC % 6 == 0)
                {
                    tC = tC / 6;
                }
                else
                {
                    tC = 1;
                }
            }
        }
        if (flag == 3)
        {
            if (tR != 1)
            {
                if (tR % 2 == 0)
                {
                    tR = tR / 2;
                }
                else if (tR % 3 == 0)
                {
                    tR = tR / 3;
                }
                else if (tR % 4 == 0)
                {
                    tR = tR / 4;
                }
                else if (tR % 5 == 0)
                {
                    tR = tR / 5;
                }
                else if (tR % 6 == 0)
                {
                    tR = tR / 6;
                }
                else
                {
                    tR = 1;
                }
            }
            else
            {
                if (tC % 2 == 0)
                {
                    tC = tC / 2;
                }
                else if (tC % 3 == 0)
                {
                    tC = tC / 3;
                }
                else if (tC % 4 == 0)
                {
                    tC = tC / 4;
                }
                else if (tC % 5 == 0)
                {
                    tC = tC / 5;
                }
                else if (tC % 6 == 0)
                {
                    tC = tC / 6;
                }
                else
                {
                    tC = 1;
                }
            }
        }
        flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
        if (flag == 0)
        {
            tR = tR * 2;
            flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
            if (flag == 0)
            {
                tR = tR * 2;
                flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
                if (flag == 0)
                {
                    tR = tR * 2;
                    flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
                }
                else
                {
                    tR = tR / 2;
                    flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
                }
            }
            else
            {
                tR = tR / 2;
                flag = calculate_buffer_consumption(N, tM, tR, tC, N, M, w->lens[2], s, pad, d);
            }
        }
    }

    tX[0] = tM;
    tX[1] = tR;
    tX[2] = tC;
    return 0;
}

///////////////////   FOR instruction generation

int output_binary(long long value, int blen)
{
    // return 0;
    while (blen--)
    {
        printf("%d", (value >> blen) & 1);
    }
}

int output_raw_binary(char *out, int *index, long long value, int blen)
{
    for (int i = 0; i < blen; i++)
    {
        out[*index / 8] &= ~(1 << (*index % 8));
        out[*index / 8] |= (((value >> i) & 1) << ((*index) % 8));
        (*index)++;
    }
}

int output_vliw_binary(FILE *fpo, struct VLIW *vliw)
{
    char out[64];
    int index = 0;
    output_binary(0, 127); // high 131 bits are currently not used
    output_binary(vliw->Conv_quant_factor2, 32);
    output_binary(vliw->Conv_quant_factor, 34);
    output_binary(vliw->Conv_permuteN, 2);
    output_binary(vliw->Conv_permuteM, 2);
    output_binary(vliw->Conv_permuteC, 2);
    output_binary(vliw->Conv_permuteR, 2);
    output_binary(vliw->Conv_tN, 12);
    output_binary(vliw->Conv_tM, 12);
    output_binary(vliw->Conv_tC, 11);
    output_binary(vliw->Conv_tR, 11);
    output_binary(vliw->Conv_dilation, 3);
    output_binary(vliw->Conv_stride, 3);
    output_binary(vliw->Conv_kernel, 5);
    output_binary(vliw->Conv_pad, 3);
    output_binary(vliw->Quant_y_z, 8);
    output_binary(vliw->Quant_x2_z, 8);
    output_binary(vliw->Quant_x1_z, 8);
    output_binary(vliw->M_concat, 12);
    output_binary(vliw->sM_concat, 12);
    output_binary(vliw->C0, 11);
    output_binary(vliw->R0, 11);
    output_binary(vliw->N, 12);
    output_binary(vliw->M, 12);
    output_binary(vliw->C, 11);
    output_binary(vliw->R, 11);
    output_binary(vliw->Activate_LUT_address, 8);
    output_binary(vliw->Compute_Result_dest_address, 36);
    output_binary(vliw->Bias_source_address, 11);
    output_binary(vliw->DDR_x2_address, 36);
    output_binary(vliw->DDR_x1_address, 36);
    output_binary(vliw->operator, 8);
    ////////
    output_raw_binary(out, &index, vliw->operator, 8);
    output_raw_binary(out, &index, vliw->DDR_x1_address, 36);
    output_raw_binary(out, &index, vliw->DDR_x2_address, 36);
    output_raw_binary(out, &index, vliw->Bias_source_address, 11);
    output_raw_binary(out, &index, vliw->Compute_Result_dest_address, 36);
    output_raw_binary(out, &index, vliw->Activate_LUT_address, 8);
    output_raw_binary(out, &index, vliw->R, 11);
    output_raw_binary(out, &index, vliw->C, 11);
    output_raw_binary(out, &index, vliw->M, 12);
    output_raw_binary(out, &index, vliw->N, 12);
    output_raw_binary(out, &index, vliw->R0, 11);
    output_raw_binary(out, &index, vliw->C0, 11);
    output_raw_binary(out, &index, vliw->sM_concat, 12);
    output_raw_binary(out, &index, vliw->M_concat, 12);
    output_raw_binary(out, &index, vliw->Quant_x1_z, 8);
    output_raw_binary(out, &index, vliw->Quant_x2_z, 8);
    output_raw_binary(out, &index, vliw->Quant_y_z, 8);
    output_raw_binary(out, &index, vliw->Conv_pad, 3);
    output_raw_binary(out, &index, vliw->Conv_kernel, 5);
    output_raw_binary(out, &index, vliw->Conv_stride, 3);
    output_raw_binary(out, &index, vliw->Conv_dilation, 3);
    output_raw_binary(out, &index, vliw->Conv_tR, 11);
    output_raw_binary(out, &index, vliw->Conv_tC, 11);
    output_raw_binary(out, &index, vliw->Conv_tM, 12);
    output_raw_binary(out, &index, vliw->Conv_tN, 12);
    output_raw_binary(out, &index, vliw->Conv_permuteR, 2);
    output_raw_binary(out, &index, vliw->Conv_permuteC, 2);
    output_raw_binary(out, &index, vliw->Conv_permuteM, 2);
    output_raw_binary(out, &index, vliw->Conv_permuteN, 2);
    output_raw_binary(out, &index, vliw->Conv_quant_factor, 34);
    output_raw_binary(out, &index, vliw->Conv_quant_factor2, 32);
    output_raw_binary(out, &index, 0, 127); // high 131 bits are currently not used

    fwrite(out, 1, 64, fpo);
}

// 新增：将VLIW结构体字段以16进制格式输出到txt文件
int output_vliw_hex(FILE *fp_hex, struct VLIW *vliw)
{
    if (fp_hex == NULL) return -1;

    fprintf(fp_hex, "operator=0x%02llx\n", vliw->operator & 0xFF);
    fprintf(fp_hex, "DDR_x1_address=0x%09llx\n", vliw->DDR_x1_address & 0xFFFFFFFFF);
    fprintf(fp_hex, "DDR_x2_address=0x%09llx\n", vliw->DDR_x2_address & 0xFFFFFFFFF);
    fprintf(fp_hex, "Bias_source_address=0x%03llx\n", vliw->Bias_source_address & 0x7FF);
    fprintf(fp_hex, "Compute_Result_dest_address=0x%09llx\n", vliw->Compute_Result_dest_address & 0xFFFFFFFFF);
    fprintf(fp_hex, "Activate_LUT_address=0x%02llx\n", vliw->Activate_LUT_address & 0xFF);
    fprintf(fp_hex, "R=0x%03llx\n", vliw->R & 0x7FF);
    fprintf(fp_hex, "C=0x%03llx\n", vliw->C & 0x7FF);
    fprintf(fp_hex, "M=0x%03llx\n", vliw->M & 0xFFF);
    fprintf(fp_hex, "N=0x%03llx\n", vliw->N & 0xFFF);
    fprintf(fp_hex, "R0=0x%03llx\n", vliw->R0 & 0x7FF);
    fprintf(fp_hex, "C0=0x%03llx\n", vliw->C0 & 0x7FF);
    fprintf(fp_hex, "sM_concat=0x%03llx\n", vliw->sM_concat & 0xFFF);
    fprintf(fp_hex, "M_concat=0x%03llx\n", vliw->M_concat & 0xFFF);
    fprintf(fp_hex, "Quant_x1_z=0x%02llx\n", vliw->Quant_x1_z & 0xFF);
    fprintf(fp_hex, "Quant_x2_z=0x%02llx\n", vliw->Quant_x2_z & 0xFF);
    fprintf(fp_hex, "Quant_y_z=0x%02llx\n", vliw->Quant_y_z & 0xFF);
    fprintf(fp_hex, "Conv_pad=0x%01llx\n", vliw->Conv_pad & 0x7);
    fprintf(fp_hex, "Conv_kernel=0x%02llx\n", vliw->Conv_kernel & 0x1F);
    fprintf(fp_hex, "Conv_stride=0x%01llx\n", vliw->Conv_stride & 0x7);
    fprintf(fp_hex, "Conv_dilation=0x%01llx\n", vliw->Conv_dilation & 0x7);
    fprintf(fp_hex, "Conv_tR=0x%03llx\n", vliw->Conv_tR & 0x7FF);
    fprintf(fp_hex, "Conv_tC=0x%03llx\n", vliw->Conv_tC & 0x7FF);
    fprintf(fp_hex, "Conv_tM=0x%03llx\n", vliw->Conv_tM & 0xFFF);
    fprintf(fp_hex, "Conv_tN=0x%03llx\n", vliw->Conv_tN & 0xFFF);
    fprintf(fp_hex, "Conv_permuteR=0x%01llx\n", vliw->Conv_permuteR & 0x3);
    fprintf(fp_hex, "Conv_permuteC=0x%01llx\n", vliw->Conv_permuteC & 0x3);
    fprintf(fp_hex, "Conv_permuteM=0x%01llx\n", vliw->Conv_permuteM & 0x3);
    fprintf(fp_hex, "Conv_permuteN=0x%01llx\n", vliw->Conv_permuteN & 0x3);
    fprintf(fp_hex, "Conv_quant_factor=0x%06llx\n", vliw->Conv_quant_factor & 0x3FFFFFFFF);
    fprintf(fp_hex, "Conv_quant_factor2=0x%06llx\n", vliw->Conv_quant_factor2 & 0xFFFFFFFF);
    fprintf(fp_hex, "---\n");
    return 0;
}

int display_buffer_consumption(int tN, int tM, int tR, int tC, int N, int M, int kernel, int stride, int pad, int dilation)
{
    int tN32_rd = 0; // rounded
    for (int sn = 0; sn < N; sn += tN)
    {
        int t = ((sn + tN + 31) >> 5) - (sn >> 5);
        if (t > tN32_rd)
            tN32_rd = t;
    }
    int tM16_rd = 0;
    for (int sm = 0; sm < M; sm += tM)
    {
        int t = ((sm + tM + 15) >> 4) - (sm >> 4);
        if (t > tM16_rd)
            tM16_rd = t;
    }
    printf("tN32_rd=%d,tM16_rd=%d\n", tN32_rd, tM16_rd);
    int relems = (tR - 1) * stride + (kernel - 1) * dilation + 1;
    int celems = (tC - 1) * stride + (kernel - 1) * dilation + 1;
    int gdepth = relems * celems * tN32_rd;
    printf("need on-chip global buffer=%d(0x%x) depth\n", gdepth, gdepth);
    int wbuf_single_m_size = tN32_rd * kernel * kernel;
    int wdepth = wbuf_single_m_size * tM16_rd;
    printf("need on-chip weight buffer=%d(0x%x) depth\n", wdepth, wdepth);
    int odepth = tR * tC * tM16_rd;
    printf("need on-chip output buffer=%d(0x%x) depth\n", odepth, odepth);

    // if (wdepth > 256 || gdepth > 1024 || odepth > 2048)
    // {
    //     system("pause");
    // }
}

// must declare WEIGHT_GEN SILU_GEN and BIAS_GEN
extern int weight_addr;
extern int silu_addr;
extern int bias_addr;

int conv_instruction_gen(FILE *fpo, FILE *fp_hex, struct TensorQ *x, struct TensorQ *w, int d, int k, int s, int pad, int tM, int tR, int tC, int sM, int Mconcat, int zx, int zw, int zo, unsigned long long source_addr, unsigned long long dest_addr, unsigned long long factor)
{
    // return 0;
    struct VLIW vliw1;
    vliw1.operator= 0;
    vliw1.DDR_x1_address = source_addr;
    vliw1.DDR_x2_address = 0x8000000 + weight_addr;
    vliw1.Bias_source_address = bias_addr;
    vliw1.Compute_Result_dest_address = dest_addr;
    vliw1.Activate_LUT_address = silu_addr;
    vliw1.R = (x->lens[2] + pad * 2 - (d * (w->lens[2] - 1) + 1)) / s + 1;
    vliw1.C = (x->lens[3] + pad * 2 - (d * (w->lens[3] - 1) + 1)) / s + 1;
    vliw1.M = w->lens[0];
    vliw1.N = w->lens[1];
    vliw1.R0 = x->lens[2];
    ;
    vliw1.C0 = x->lens[3];
    vliw1.sM_concat = sM;
    vliw1.M_concat = Mconcat;
    printf("conv:R,C,M,N,R0,C0,SM,MCONCAT %d,%d,%d,%d,%d,%d,%d,%d\n",
           vliw1.R, vliw1.C, vliw1.M, vliw1.N, vliw1.R0, vliw1.C0, vliw1.sM_concat, vliw1.M_concat);
    vliw1.Quant_x1_z = zx;
    vliw1.Quant_x2_z = zw;
    vliw1.Quant_y_z = zo;
    vliw1.Conv_pad = pad;
    vliw1.Conv_kernel = k;
    vliw1.Conv_stride = s;
    vliw1.Conv_dilation = d;
    vliw1.Conv_tR = tR;
    vliw1.Conv_tC = tC;
    vliw1.Conv_tM = tM;
    vliw1.Conv_tN = x->lens[1];
    vliw1.Conv_permuteR = 0;
    vliw1.Conv_permuteC = 0;
    vliw1.Conv_permuteM = 0;
    vliw1.Conv_permuteN = 0;
    vliw1.Conv_quant_factor = factor;
    output_vliw_binary(fpo, &vliw1);
    output_vliw_hex(fp_hex, &vliw1);
    printf("\n");
    display_buffer_consumption(vliw1.Conv_tN, vliw1.Conv_tM, vliw1.Conv_tR, vliw1.Conv_tC, vliw1.N, vliw1.M, vliw1.Conv_kernel, vliw1.Conv_stride, vliw1.Conv_pad, vliw1.Conv_dilation);
}

int depthconv_instruction_gen(FILE *fpo, FILE *fp_hex, struct TensorQ *x, struct TensorQ *w, int d, int k, int s, int pad, int tM, int tR, int tC, int sM, int Mconcat, int zx, int zw, int zo, unsigned long long source_addr, unsigned long long dest_addr, unsigned long long factor)
{
    struct VLIW vliw1;
    vliw1.operator= 0;
    vliw1.DDR_x1_address = source_addr;             // fmap
    vliw1.DDR_x2_address = 0x8000000 + weight_addr; // weight
    vliw1.Bias_source_address = bias_addr;          // bias
    vliw1.Compute_Result_dest_address = dest_addr;  // res
    // printf("addr: %lu\n", vliw1.Compute_Result_dest_address);
    vliw1.Activate_LUT_address = silu_addr; // activate
    vliw1.R = (x->lens[2] + pad * 2 - (1) * w->lens[2]) / s + 1;
    ;                                                            // 经过s和p后的R
    vliw1.C = (x->lens[3] + pad * 2 - (1) * w->lens[3]) / s + 1; // 经过s和p后的C
    vliw1.M = w->lens[0];                                        // M输出通道数cout
    vliw1.N = x->lens[1];                                        // N输入通道数cin，分组卷积这里是1，我不能让他是1，是1就寄了
    vliw1.R0 = x->lens[2];
    ;                         // 原本的不pad的R0
    vliw1.C0 = x->lens[3];    // 原本的不pad的C0
    vliw1.sM_concat = sM;     // 起始位置
    vliw1.M_concat = Mconcat; // 为什么是32，因为通道就是32，那么拼接后也是32
    printf("R,C,M,N,R0,C0,SM,SMCONCAT %d,%d,%d,%d,%d,%d,%d,%d\n",
           vliw1.R, vliw1.C, vliw1.M, vliw1.N, vliw1.R0, vliw1.C0, vliw1.sM_concat, vliw1.M_concat);
    vliw1.Quant_x1_z = zx;
    vliw1.Quant_x2_z = zw;
    vliw1.Quant_y_z = zo;
    vliw1.Conv_pad = pad;
    vliw1.Conv_kernel = k;
    vliw1.Conv_stride = s;
    vliw1.Conv_dilation = 1;
    vliw1.Conv_tR = tR;
    vliw1.Conv_tC = tC;
    vliw1.Conv_tM = tM;
    vliw1.Conv_tN = tM;
    vliw1.Conv_permuteR = 1;
    vliw1.Conv_permuteC = 2;
    vliw1.Conv_permuteM = 1;
    vliw1.Conv_permuteN = 0;
    vliw1.Conv_quant_factor = factor;
    output_vliw_binary(fpo, &vliw1);
    output_vliw_hex(fp_hex, &vliw1);
    printf("\n");
    display_buffer_consumption(vliw1.Conv_tN, vliw1.Conv_tM, vliw1.Conv_tR, vliw1.Conv_tC, vliw1.N, vliw1.M, vliw1.Conv_kernel, vliw1.Conv_stride, vliw1.Conv_pad, vliw1.Conv_dilation);
}

int res_instruction_gen(FILE *fpo, FILE *fp_hex, struct TensorQ *x, struct TensorQ *w, int tM, int tR, int tC, int sM, int Mconcat, int yz, int xz, int wz, unsigned long long source_addr, unsigned long long source_addr2, unsigned long long dest_addr, long long factor, long long factor2)
{
    // return 0;
    struct VLIW vliw1;
    vliw1.operator= 3;
    vliw1.DDR_x1_address = source_addr;
    vliw1.DDR_x2_address = source_addr2;
    vliw1.Bias_source_address = bias_addr;
    vliw1.Compute_Result_dest_address = dest_addr;
    vliw1.Activate_LUT_address = silu_addr;
    vliw1.R = x->lens[2];
    vliw1.C = x->lens[3];
    vliw1.M = x->lens[1];
    vliw1.N = x->lens[1];
    vliw1.R0 = x->lens[2];
    vliw1.C0 = x->lens[3];
    vliw1.sM_concat = sM;
    vliw1.M_concat = Mconcat;
    vliw1.Quant_y_z = yz;
    vliw1.Quant_x1_z = xz;
    vliw1.Quant_x2_z = wz;
    printf("res:R,C,M,N,R0,C0,SM,SMCONCAT %d,%d,%d,%d,%d,%d,%d,%d\n",
           vliw1.R, vliw1.C, vliw1.M, vliw1.N, vliw1.R0, vliw1.C0, vliw1.sM_concat, vliw1.M_concat);
    vliw1.Conv_pad = 0;
    vliw1.Conv_kernel = 1;
    vliw1.Conv_stride = 1;
    vliw1.Conv_dilation = 1;
    vliw1.Conv_tR = tR;
    vliw1.Conv_tC = tC;
    vliw1.Conv_tM = x->lens[1];
    vliw1.Conv_tN = tM;
    vliw1.Conv_permuteR = 0;
    vliw1.Conv_permuteC = 0;
    vliw1.Conv_permuteM = 0;
    vliw1.Conv_permuteN = 0;
    vliw1.Conv_quant_factor = factor;
    vliw1.Conv_quant_factor2 = factor2;
    output_vliw_binary(fpo, &vliw1);
    output_vliw_hex(fp_hex, &vliw1);
    printf("\n");
    display_buffer_consumption(vliw1.Conv_tN, vliw1.Conv_tN, vliw1.Conv_tR, vliw1.Conv_tC, vliw1.N, vliw1.M, vliw1.Conv_kernel, vliw1.Conv_stride, vliw1.Conv_pad, vliw1.Conv_dilation);
}

int mpool_instruction_gen(FILE *fpo, FILE *fp_hex, struct TensorQ *x, int tM, int tR, int tC, int pad, int kernel, int stride, int sM, int Mconcat, unsigned long long source_addr, unsigned long long dest_addr)
{
    // return 0;
    struct VLIW vliw1;
    vliw1.operator= 1;
    vliw1.DDR_x1_address = source_addr;
    vliw1.Bias_source_address = bias_addr;
    vliw1.Compute_Result_dest_address = dest_addr;
    vliw1.Activate_LUT_address = silu_addr;
    vliw1.R = (x->lens[2] + pad * 2 - kernel) / stride + 1;
    vliw1.C = (x->lens[3] + pad * 2 - kernel) / stride + 1;
    vliw1.M = x->lens[1];
    vliw1.N = x->lens[1];
    vliw1.R0 = x->lens[2];
    vliw1.C0 = x->lens[3];
    vliw1.sM_concat = sM;
    vliw1.M_concat = Mconcat;
    vliw1.Quant_x1_z = -128;
    printf("mpool:R,C,M,N,R0,C0,SM,SMCONCAT %d,%d,%d,%d,%d,%d,%d,%d\n",
           vliw1.R, vliw1.C, vliw1.M, vliw1.N, vliw1.R0, vliw1.C0, vliw1.sM_concat, vliw1.M_concat);
    vliw1.Conv_pad = pad;
    vliw1.Conv_kernel = kernel;
    vliw1.Conv_stride = stride;
    vliw1.Conv_dilation = 1;
    vliw1.Conv_tR = tR;
    vliw1.Conv_tC = tC;
    vliw1.Conv_tM = x->lens[1];
    vliw1.Conv_tN = tM; // in this mode we do not iterate M,each iteration fetch data <tR,tC,tN> into global buffer
    vliw1.Conv_permuteR = 0;
    vliw1.Conv_permuteC = 0;
    vliw1.Conv_permuteM = 0;
    vliw1.Conv_permuteN = 0;
    output_vliw_binary(fpo, &vliw1);
    output_vliw_hex(fp_hex, &vliw1);
    printf("\n");

    // in this mode,we only care the global buffer consumption and output buffer consumption
    //  the weight buffer consumption is zero
    display_buffer_consumption(vliw1.Conv_tN, vliw1.Conv_tN, vliw1.Conv_tR, vliw1.Conv_tC, vliw1.N, vliw1.M, vliw1.Conv_kernel, vliw1.Conv_stride, vliw1.Conv_pad, vliw1.Conv_dilation);
}

int usample_instruction_gen(FILE *fpo, FILE *fp_hex, struct TensorQ *x, int tM, int tR, int tC, int sM, int Mconcat, unsigned long long source_addr, unsigned long long dest_addr)
{
    // return 0;
    struct VLIW vliw1;
    vliw1.operator= 2;
    vliw1.DDR_x1_address = source_addr;
    vliw1.Bias_source_address = bias_addr;
    vliw1.Compute_Result_dest_address = dest_addr;
    vliw1.Activate_LUT_address = silu_addr;
    vliw1.R = x->lens[2];
    vliw1.C = x->lens[3];
    vliw1.M = x->lens[1];
    vliw1.N = x->lens[1];
    vliw1.R0 = x->lens[2];
    vliw1.C0 = x->lens[3];
    vliw1.sM_concat = sM;
    vliw1.M_concat = Mconcat;
    vliw1.Quant_x1_z = -128;
    printf("R,C,M,N,R0,C0,SM,SMCONCAT %d,%d,%d,%d,%d,%d,%d,%d\n",
           vliw1.R, vliw1.C, vliw1.M, vliw1.N, vliw1.R0, vliw1.C0, vliw1.sM_concat, vliw1.M_concat);
    vliw1.Conv_pad = 0;
    vliw1.Conv_kernel = 1;
    vliw1.Conv_stride = 1;
    vliw1.Conv_dilation = 1;
    vliw1.Conv_tR = tR;
    vliw1.Conv_tC = tC;
    vliw1.Conv_tM = x->lens[1];
    vliw1.Conv_tN = tM;
    vliw1.Conv_permuteR = 0;
    vliw1.Conv_permuteC = 0;
    vliw1.Conv_permuteM = 0;
    vliw1.Conv_permuteN = 0;
    output_vliw_binary(fpo, &vliw1);
    output_vliw_hex(fp_hex, &vliw1);
    printf("\n");
    display_buffer_consumption(vliw1.Conv_tN, vliw1.Conv_tM, vliw1.Conv_tR, vliw1.Conv_tC, vliw1.N, vliw1.M, vliw1.Conv_kernel, vliw1.Conv_stride, vliw1.Conv_pad, vliw1.Conv_dilation);
}

// the ending instruction
int test_ff(FILE *fpo, long long address, long long length)
{
    struct VLIW vliw1;
    vliw1.operator= 0xff;
    vliw1.DDR_x1_address = address;
    // vliw1.DDR_x2_address= /* 20*20*512 */   204800+51200+12800  /* 1024*1024*2 */;  //268800
    vliw1.DDR_x2_address = length;
    output_vliw_binary(fpo, &vliw1);
}

int transmit_instruction_gen(FILE *fpo, int address, int length)
{
    struct VLIW vliw1;
    vliw1.operator= 0xff;
    vliw1.DDR_x1_address = address;
    vliw1.DDR_x2_address = length;
    output_vliw_binary(fpo, &vliw1);
}

//------IEEE754-----------

void processFractionalPart(uint32_t fractional_part, uint32_t *fractional_parts)
{
    for (int i = 0; i < 32; i++)
    {
        fractional_part <<= 1;                       // 左移一位
        fractional_parts[i] = fractional_part >> 23; // 取整数部分（最高23位）
        fractional_part &= 0x7FFFFF;                 // 获取剩余的尾数部分（低23位）
    }
}

void convertToIEEE754(uint32_t sign, uint32_t integer_part, uint32_t fractional_part, uint32_t *result_final)
{
    uint32_t integer_part_bk = integer_part;

    // Step 1: Process fractional part
    uint32_t fractional_parts[32];
    processFractionalPart(fractional_part, fractional_parts);

    if (sign == 0 && integer_part == 0 && fractional_part == 0)
    { // new add
        *result_final = 0;
    }
    else if (integer_part == 0)
    {
        int position = 0;
        for (int i = 0; i < 32; i++)
        {
            if (fractional_parts[i] == 1)
            {
                position = i;
                break;
            }
        }
        int exponent = 127 - (position + 1);

        int significand_32_update[32];
        int start_nozero = position;
        for (int j = 0; j < 32; j++)
        {
            significand_32_update[j] = fractional_parts[start_nozero];
            start_nozero++;
        }

        // Step 7: Combine the final IEEE 754 result (sign, exponent, significand)
        uint32_t result = 0;
        for (int i = 0; i < 32; i++)
        {
            result |= (significand_32_update[i] << (31 - i));
        }

        // Step 8: Finalize the IEEE 754 representation
        uint32_t sign_final = sign; // Default to 0 for positive numbers, update if needed
        uint32_t exponent_final = exponent;
        uint32_t significand_final = (result << 1); // Shift for the leading 1 in the normalized form
        unsigned int mask = 0x7FFFFF;               // Mask to get the 23 bits of the significand

        // Combining the final IEEE 754 result
        *result_final = (sign_final << 31) | (exponent_final << 23) | ((significand_final >> 9) & mask);
    }
    else
    {
        // Step 2: Determine exponent for scientific notation
        int position = 0;
        int exponent_sci = 0;
        while (integer_part > 0)
        {
            if (integer_part & 1)
            {
                exponent_sci = position; // Find the position of the most significant 1 bit
            }
            integer_part >>= 1; // Shift right to check the next bit
            position++;
        }

        // Step 3: Extract integer part bits (8 bits)
        int integer_part_bits[8];
        int k = 7; // We start filling from the highest bit position
        for (int i = 0; i < 8; i++)
        {
            integer_part_bits[k] = (integer_part_bk >> i) & 1;
            k--;
        }

        // Step 4: Set exponent for IEEE 754 format
        int exponent = exponent_sci + 127; // Bias of 127 for single precision (32-bit float)

        // Step 5: Combine the integer part bits and fractional part bits into the significand (32 bits)
        int Significand_32[40];
        for (int i = 0; i < 8 + 32; i++)
        {
            if (i < 8)
            {
                Significand_32[i] = integer_part_bits[i];
            }
            else
            {
                Significand_32[i] = fractional_parts[i - 8];
            }
        }

        // if (integer_part_bk == 203)
        // {
        //     for (int i = 0; i < 40; i++)
        //     {
        //         printf("%d", Significand_32[i]);
        //     }
        //     printf("\n");
        // }

        // Step 6: Update significand to remove leading zeros (if any) for proper normalization
        int significand_32_update[32];
        int start_nozero = (8 - (exponent_sci + 1));
        for (int j = 0; j < 32; j++)
        {
            significand_32_update[j] = Significand_32[start_nozero];
            start_nozero++;
        }

        if (significand_32_update[24])
        {
            int carry = 1;
            for (int i = 24; i >= 0 && carry; i--)
            {
                significand_32_update[i] += carry;
                if (significand_32_update[i] > 1)
                {
                    significand_32_update[i] = 0;
                    carry = 1;
                }
                else
                {
                    carry = 0;
                }
            }
        }
        // else:
        // Step 7: Combine the final IEEE 754 result (sign, exponent, significand)
        uint32_t result = 0;
        for (int i = 0; i < 32; i++)
        {
            result |= (significand_32_update[i] << (31 - i));
        }

        // Step 8: Finalize the IEEE 754 representation
        uint32_t sign_final = sign; // Default to 0 for positive numbers, update if needed
        uint32_t exponent_final = exponent;
        uint32_t significand_final = (result << 1); // Shift for the leading 1 in the normalized form
        unsigned int mask = 0x7FFFFF;               // Mask to get the 23 bits of the significand

        // Combining the final IEEE 754 result
        *result_final = (sign_final << 31) | (exponent_final << 23) | ((significand_final >> 9) & mask);
    }
}

int my_nearbyint(uint32_t x)
{
    // 提取符号位、指数位和尾数位
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exponent = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;

    // 处理特殊情况：NaN 或无穷大
    if (exponent == 128)
    {
        printf("NaN happend!\n");
        system("pause");
        return x;
    }

    // 如果指数为负，说明 |x| < 1.0

    if (exponent < 0)
    {
        if (exponent == -127 && mantissa == 0)
        {
            return 0;
        }
        if (exponent == -1 && mantissa != 0)
        {
            return sign == 0 ? 1 : -1;
        }
        return 0;
    }

    // 有待商榷
    //  if (exponent < 0)
    //  {
    //      if (exponent == -127 && mantissa == 0)
    //      { // 特殊情况，处理为0
    //          return 0;
    //      }
    //      if (sign == 0)
    //      { // x > 0
    //          if (exponent == -1)
    //          { //|x| >= 0.5
    //              if (mantissa == 0)
    //              {
    //                  return 0;
    //              }
    //              else
    //              {
    //                  return 1;
    //              }
    //          }
    //          else
    //          { // |x| < 0.5
    //              return 0;
    //          }
    //      }
    //      else
    //      { // x <0
    //          if (exponent == -1)
    //          { //|x| >= 0.5
    //              if (mantissa == 0)
    //              {
    //                  return 0;
    //              }
    //              else
    //              {
    //                  return -1;
    //              }
    //          }
    //          else
    //          { // |x| < 0.5
    //              return 0;
    //          }
    //      }
    //  }

    // 补全尾数的前导 1（除非是次正规数）
    if (exponent != -127)
    {
        mantissa |= 0x800000; // 补全前导 1
    }

    // 计算尾数的整数部分和小数部分
    uint32_t shift = 23 - exponent;
    // if (shift > 23)
    // {
    //     return x;
    // }

    uint32_t integer_part = (mantissa >> shift);
    uint32_t fractional_part = mantissa & ((1 << shift) - 1);

    // 处理向偶数舍入
    if (fractional_part > (1 << (shift - 1)))
    {
        // 如果尾数大于当前舍入位的中间值，直接向上舍入
        integer_part++;
    }
    else if (fractional_part == (1 << (shift - 1)))
    {
        // 如果尾数恰好等于中间值，进行“向偶数舍入”
        if (integer_part % 2 != 0)
        {
            integer_part++;
        }
    }

    // 处理符号位
    if (sign)
    {
        integer_part = -integer_part;
    }

    // 将整数部分转换回浮点数
    return (int)integer_part;
}

int result_compare(int x0, int x1, int x2, int x3, struct TensorQ *x, const char *filename)
{
    // if ((x0 != x->lens[0]) || (x1 != x->lens[1]) || (x2 != x->lens[2]) || (x3 != x->lens[3]))
    //     printf("compare faile, due data type not match.\n");
    //     return 0;
    struct TensorQ *onnx_result = getTensorQ(4);
    onnx_result->lens[0] = x0;
    onnx_result->lens[1] = x1;
    onnx_result->lens[2] = x2;
    onnx_result->lens[3] = x3;
    // onnx_result->data=(int*)readfile(filename);
    // onnx_result->data=(unsigned char *)readfile(filename);

    char *file_data = (char *)readfile(filename);

    int precision = 4;
    int error = 0;
    char str1[20], str2[20];
    for (int i = 0; i < x0; i++)
    {
        for (int j = 0; j < x1; j++)
        {
            if (j == 2)
            {
                printf("attention.\n");
            }
            for (int k = 0; k < x2; k++)
            {
                for (int m = 0; m < x3; m++)
                {
                    // int onnx_data = onnx_result->data[getindex(4, i, j, k, m, onnx_result->lens)];
                    int onnx_data = (int *)file_data[getindex(4, i, j, k, m, onnx_result->lens)];
                    int c_data = x->data[getindex(4, i, j, k, m, x->lens)];

                    if (onnx_data != c_data)
                    {
                        // if (abs(onnx_data - c_data) > 5){
                        //     error++;
                        //     printf("[result compare error] %d, %d, %d, %d, onnx_result=%d, c_data=%d.\n", i, j , k, m, onnx_data, c_data);
                        // //assert(0);
                        // }
                        error++;
                        printf("[result compare error] %d, %d, %d, %d, onnx_result=%d, c_data=%d.\n", i, j, k, m, onnx_data, c_data);
                    }
                }
            }
        }
    }
    if (error != 0)
        printf("compare faile.\n");
    return error;
}