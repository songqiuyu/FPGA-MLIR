#include "../tensor.h"
#include "../basic.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>

// we extrapolate kernel shape from tensor w
// assume dilation applies to both x and y axis
// assume pads applies to both x and y axis
// assume stride applies to both x and y axis
// assume we have only one group,i,e,group=1
struct Tensor *conv(struct Tensor *x, struct Tensor *w, struct Tensor *b, int dilation, int pads, int stride)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // w should have 4 dims
    if (w->ndim != 4)
        return 0;
    // b should have 1 dims
    if (b->ndim != 1)
        return 0;
    // b dim0 should equal to w dim0
    if (b->lens[0] != w->lens[0])
        return 0;
    // w dim1 should equal to x dim1
    if (w->lens[1] != x->lens[1])
        return 0;
    // the shape of the result tensor is:
    struct Tensor *result = getTensor(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = w->lens[0];
    result->lens[2] = (x->lens[2] + pads * 2 - dilation * w->lens[2]) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - dilation * w->lens[3]) / stride + 1;
    result->data = (DTYPE *)malloc(sizeof(DTYPE) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);
    // the result tensor is:
#pragma omp parallel for

    for (int i = 0; i < w->lens[0]; i++)
    { // for each output feature maps,indexed by i
      // printf("i=%d\n",i);
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++)
            { // for each output point,indexed by (j,k)
              // printf("j=%d,k=%d\n",j,k);

                DTYPE dotproduct = 0;
                for (int l = 0; l < w->lens[1]; l++)
                {
                    for (int m = 0; m < w->lens[2]; m++)
                    {
                        for (int n = 0; n < w->lens[3]; n++)
                        { // for each kernel point,indexed by (l,m,n)
                            // printf("l=%d,m=%d,n=%d\n",l,m,n);
                            DTYPE xdat = (j * stride + m * dilation < pads || j * stride + m * dilation >= pads + x->lens[2] || k * stride + n * dilation < pads || k * stride + n * dilation >= pads + x->lens[3]) ? 0 : x->data[getindex(4, 0, l, j * stride + m * dilation - pads, k * stride + n * dilation - pads, x->lens)];
                            // printf("position at %d,%d xdat=%f\n",j*stride+m*dilation,k*stride+n*dilation,xdat);
                            dotproduct += w->data[getindex(4, i, l, m, n, w->lens)] * xdat; //
                        }
                    }
                }
                // data at result(0,i,j,k) should be the dotproduct plus the i th bias
                result->data[getindex(4, 0, i, j, k, result->lens)] = dotproduct + b->data[i];
            }
        }
    }

    return result;
}

// optimized version
struct Tensor *conv2(struct Tensor *x, struct Tensor *w, struct Tensor *b, int dilation, int pads, int stride)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // w should have 4 dims
    if (w->ndim != 4)
        return 0;
    // b should have 1 dims
    if (b->ndim != 1)
        return 0;
    // b dim0 should equal to w dim0
    if (b->lens[0] != w->lens[0])
        return 0;
    // w dim1 should equal to x dim1
    if (w->lens[1] != x->lens[1])
        return 0;
    // the shape of the result tensor is:
    struct Tensor *result = getTensor(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = w->lens[0];
    result->lens[2] = (x->lens[2] + pads * 2 - dilation * w->lens[2]) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - dilation * w->lens[3]) / stride + 1;
    result->data = (DTYPE *)malloc(sizeof(DTYPE) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);
    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < w->lens[0]; i++)
    { // for each output feature map,indexed by i
      // printf("i=%d\n",i);
        int length = getlength(x) + ((pads + x->lens[2]) * pads * 2 + (pads + x->lens[3]) * pads * 2) * x->lens[1];
        // produce temp1 buffer
        DTYPE *temp1 = (DTYPE *)malloc(sizeof(DTYPE) * length);
        memset(temp1, 0, length * sizeof(DTYPE));
        // copy the data
        int dims[3];
        dims[0] = x->lens[1];
        dims[1] = x->lens[2] + pads * 2;
        dims[2] = x->lens[3] + pads * 2;
        int indexsrc = 0;
        for (int f = 0; f < x->lens[1]; f++)
        {
            int indexdst = getindex(3, f, 0 + pads, 0 + pads, dims);
            for (int j = 0; j < x->lens[2]; j++)
            {
                memcpy(temp1 + indexdst, x->data + indexsrc, sizeof(DTYPE) * x->lens[3]);
                indexdst += dims[2];
                indexsrc += x->lens[3];
            }
        }
        int length2 = result->lens[2] * result->lens[3];    // result length
        int kernlen = w->lens[1] * w->lens[2] * w->lens[3]; // kernel size
        // slice the data into temp2
        DTYPE *temp2 = (DTYPE *)malloc(sizeof(DTYPE) * length2 * kernlen);
        memset(temp2, 0, length2 * kernlen * sizeof(DTYPE));
        int index2 = 0;
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++)
            {
                int baseindex = j * stride * dims[2] + k * stride;
                for (int l = 0; l < w->lens[1]; l++)
                {
                    int baseindexl = baseindex;
                    for (int m = 0; m < w->lens[2]; m++)
                    {
                        int baseindexm = baseindex;
                        for (int n = 0; n < w->lens[3]; n++)
                        {
                            temp2[index2++] = temp1[baseindex];
                            baseindex += dilation;
                        }
                        baseindex = baseindexm + dims[2] * dilation;
                    }
                    baseindex = baseindexl + dims[1] * dims[2];
                }
            }
        }
        // finally,compute
        index2 = 0;
        int wbase = getindex(4, i, 0, 0, 0, w->lens);
        int rbase = getindex(4, 0, i, 0, 0, result->lens);
        for (int j = 0; j < length2; j++)
        {
            DTYPE dotproduct = 0;
            for (int k = 0; k < kernlen; k++)
            {
                dotproduct += temp2[index2++] * w->data[wbase + k];
            }
            result->data[rbase + j] = dotproduct + b->data[i];
        }
    }

    return result;
}

// optimized version
struct TensorQ *conv2Q(struct TensorQ *x, struct TensorQ *w, struct TensorQ *b, int dilation, int pads, int stride, int zx, int zw, int zo, long long factor, int pown)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // w should have 4 dims
    if (w->ndim != 4)
        return 0;
    // b should have 1 dims
    if (b->ndim != 1)
        return 0;
    // b dim0 should equal to w dim0
    if (b->lens[0] != w->lens[0])
        return 0;
    // w dim1 should equal to x dim1
    if (w->lens[1] != x->lens[1])
        return 0;

    // the shape of the result tensor is:
    struct TensorQ *result = getTensorQ(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = w->lens[0];
    result->lens[2] = (x->lens[2] + pads * 2 - dilation * w->lens[2]) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - dilation * w->lens[3]) / stride + 1;
    result->data = (int *)malloc(sizeof(int) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);
    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < w->lens[0]; i++)
    { // for each output feature map,indexed by i
      // printf("i=%d\n",i);
        int length = getlengthQ(x) + ((pads + x->lens[2]) * pads * 2 + (pads + x->lens[3]) * pads * 2) * x->lens[1];
        // produce temp1 buffer
        int *temp1 = (int *)malloc(sizeof(int) * length);
        for (int t = 0; t < length; t++)
        {
            temp1[t] = zx;
        }
        // copy the data
        int dims[3];
        dims[0] = x->lens[1];
        dims[1] = x->lens[2] + pads * 2;
        dims[2] = x->lens[3] + pads * 2;
        int indexsrc = 0;
        for (int f = 0; f < x->lens[1]; f++)
        {
            int indexdst = getindex(3, f, 0 + pads, 0 + pads, dims);
            for (int j = 0; j < x->lens[2]; j++)
            {
                memcpy(temp1 + indexdst, x->data + indexsrc, sizeof(int) * x->lens[3]);
                indexdst += dims[2];
                indexsrc += x->lens[3];
            }
        }
        int length2 = result->lens[2] * result->lens[3];    // result length
        int kernlen = w->lens[1] * w->lens[2] * w->lens[3]; // kernel size
        // slice the data into temp2
        int *temp2 = (int *)malloc(sizeof(int) * length2 * kernlen);
        memset(temp2, 0, length2 * kernlen * sizeof(int));
        int index2 = 0;
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++)
            {
                int baseindex = j * stride * dims[2] + k * stride;
                for (int l = 0; l < w->lens[1]; l++)
                {
                    int baseindexl = baseindex;
                    for (int m = 0; m < w->lens[2]; m++)
                    {
                        int baseindexm = baseindex;
                        for (int n = 0; n < w->lens[3]; n++)
                        {
                            temp2[index2++] = temp1[baseindex];
                            baseindex += dilation;
                        }
                        baseindex = baseindexm + dims[2] * dilation;
                    }
                    baseindex = baseindexl + dims[1] * dims[2];
                }
            }
        }
        // finally,compute
        index2 = 0;
        int wbase = getindex(4, i, 0, 0, 0, w->lens);
        int rbase = getindex(4, 0, i, 0, 0, result->lens);
        for (int j = 0; j < length2; j++)
        {
            int dotproduct = 0;
            for (int k = 0; k < kernlen; k++)
            {
                dotproduct += (temp2[index2++] - zx) * (w->data[wbase + k] - zw);
                // printf("x=%d,w=%d,dotproduct=%d\n", temp2[index2], w->data[wbase + k],dotproduct);
            }
            result->data[rbase + j] = dotproduct + b->data[i]; // has <sxsw,0>
            result->data[rbase + j] = result_product_gen_32bit(result->data[rbase + j], factor, zo);
            // result->data[rbase+j]=factor_product(result->data[rbase+j],factor,pown)+zo;
            // long long data_t = result->data[rbase + j] * factor;
            // uint32_t sign = (data_t >= 0) ? 0 : 1;
            // data_t = data_t < 0 ? -data_t : data_t;
            // int32_t data_integer_part = data_t >> 32;
            // uint32_t data_fractional_part = data_t & 0xFFFFFFFF;
            // data_fractional_part = (data_fractional_part >> 9);
            // uint32_t data_ieee754;
            // convertToIEEE754(sign, data_integer_part, data_fractional_part, &data_ieee754);
            // int data_nearbyint = my_nearbyint(data_ieee754);
            // int result_new_0311 = data_nearbyint + zo;
            // result->data[rbase + j] = result_new_0311;
        }
        free(temp1);
        free(temp2);
    }

    return result;
}

// struct TensorQ *conv2Q(struct TensorQ *x, struct TensorQ *w, struct TensorQ *b, int dilation, int pads, int stride, int zx, int zw, int zo, long long factor, int pown){
//     if (x->ndim != 4)
//         return 0;
//     // w should have 4 dims
//     if (w->ndim != 4)
//         return 0;
//     // b should have 1 dims
//     if (b->ndim != 1)
//         return 0;
//     // b dim0 should equal to w dim0
//     if (b->lens[0] != w->lens[0])
//         return 0;
//     // w dim1 should equal to x dim1
//     if (w->lens[1] != x->lens[1])
//         return 0;

//     // the shape of the result tensor is:
//     struct TensorQ *result = getTensorQ(x->ndim);
//     result->lens[0] = x->lens[0];
//     result->lens[1] = w->lens[0];
//     result->lens[2] = (x->lens[2] + pads * 2 - dilation * w->lens[2]) / stride + 1;
//     result->lens[3] = (x->lens[3] + pads * 2 - dilation * w->lens[3]) / stride + 1;
//     result->data = (int *)malloc(sizeof(int) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);

// #pragma omp parallel
//     for (int i = 0; i < result->lens[1]; i++)                // channel, group
//     {
//         #pragma omp for
//         for(int j=0;j<result->lens[2];j++){
//             for (int k = 0; k < result->lens[3]; k++)   // width
//             {
//                 int dotproduct = 0;
//                 int wdata_sum = 0;

//                 for (int l = 0; l < w->lens[1]; l++)    //
//                 {
//                     for (int m = 0; m < w->lens[2]; m++)
//                     {
//                         for (int n = 0; n < w->lens[3]; n++)
//                         {
//                             int xdat = (j * stride + m * dilation < pads ||
//                                         j * stride + m * dilation >= pads + x->lens[2] ||
//                                         k * stride + n * dilation < pads ||
//                                         k * stride + n * dilation >= pads + x->lens[3]) ? zx :
//                                         x->data[getindex(4, 0, l, j * stride + m * dilation - pads, k * stride + n * dilation - pads, x->lens)];

//                             int wdata = w->data[getindex(4, i, l, m, n, w->lens)];
//                             wdata_sum += wdata;
//                             dotproduct += (wdata*xdat);
//                         }
//                     }
//                 }

//                 int q_bias = b->data[i] - zx*wdata_sum;
//                 //Y = WX + B
//                 long long result_data = dotproduct + q_bias;
//                 result->data[getindex(4, 0, i, j, k, result->lens)] = factor_product(result_data,factor,pown)+zo;
//                 if(j==0 && k==80){
//                     printf("bias=%d\n",b->data[i]);
//                     printf("factor=%lld\n", factor);
//                     printf("pown%d\n", pown);
//                     printf("result_data=%d\n", result_data);
//                     factor_product(result_data,factor,pown)+zo;
//                     printf("zo=%d\n", zo);
//                     printf("result=%d\n\n", result->data[getindex(4, 0, i, j, k, result->lens)] = factor_product(result_data,factor,pown)+zo);
//                 }
//                 // if(i==0 && j==1 && k==28){
//                 //     printf("[debug]result=%d\n", result_data);
//                 //     int tt = factor_product(result_data,factor,pown)+zo;
//                 //     printf("[debug]result_quant=%x\n",tt);
//                 // }
//             }
//         }
//     }
//     return result;
// }

struct TensorQ *depthwiseconv2Q(struct TensorQ *x, struct TensorQ *w, struct TensorQ *b, int dilation, int pads, int stride, int zx, int zw, int zo, long long factor, int pown)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // w should have 4 dims
    if (w->ndim != 4)
        return 0;
    // b should have 1 dims
    if (b->ndim != 1)
        return 0;
    // b dim0 should equal to w dim0
    if (b->lens[0] != w->lens[0])
        return 0;

    // the shape of the result tensor is:
    struct TensorQ *result = getTensorQ(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = w->lens[0];
    result->lens[2] = (x->lens[2] + pads * 2 - dilation * (w->lens[2] - 1) - 1) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - dilation * (w->lens[3] - 1) - 1) / stride + 1;
    result->data = (int *)malloc(sizeof(int) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);

    // return result;
#pragma omp parallel
    for (int i = 0; i < result->lens[1]; i++) // channel
    {
#pragma omp for
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++) // width
            {
                int dotproduct = 0;
                int wdata_sum = 0;
                for (int l = 0; l < w->lens[1]; l++) //
                {
                    for (int m = 0; m < w->lens[2]; m++)
                    {
                        for (int n = 0; n < w->lens[3]; n++)
                        {
                            int xdat = (j * stride + m * dilation < pads ||
                                        j * stride + m * dilation >= pads + x->lens[2] ||
                                        k * stride + n * dilation < pads ||
                                        k * stride + n * dilation >= pads + x->lens[3])
                                           ? zx
                                           : x->data[getindex(4, 0, i, j * stride + m * dilation - pads, k * stride + n * dilation - pads, x->lens)];
                            int wdata = w->data[getindex(4, i, l, m, n, w->lens)];
                            wdata_sum += wdata;
                            dotproduct += (wdata * xdat);
                            // if(i==7 && j==0 && k==0){
                            //     printf("[%d x %d = %d]\n", xdat, wdata, dotproduct);
                            // }
                        }
                    }
                }
                int q_bias = b->data[i] - zx * wdata_sum; // Q_bias = Qb - ZxQw
                long long result_data = dotproduct + q_bias;
                //result->data[getindex(4, 0, i, j, k, result->lens)] = factor_product(result_data, factor, pown) + zo;
                // result->data[getindex(4, 0, i, j, k, result->lens)] = result_product_gen_32bit(result_data, factor, zo);
                result->data[getindex(4, 0, i, j, k, result->lens)] = result_product_gen_32bit(result_data, factor, zo);
                // long long data_t = result_data * factor;
                // uint32_t sign = (data_t >= 0) ? 0 : 1;
                // data_t = data_t < 0 ? -data_t : data_t;
                // int32_t data_integer_part = data_t >> 32;
                // uint32_t data_fractional_part = data_t & 0xFFFFFFFF;
                // data_fractional_part = (data_fractional_part >> 9);
                // uint32_t data_ieee754;
                // convertToIEEE754(sign, data_integer_part, data_fractional_part, &data_ieee754);
                // int data_nearbyint = my_nearbyint(data_ieee754);
                // int result_new_0311 = data_nearbyint + zo;
                // result->data[getindex(4, 0, i, j, k, result->lens)] = result_new_0311;
            }
        }
    }
    return result;
}