#include "../tensor.h"
#include "../basic.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

// assume dilation=1
// assume ceil mode=0
// kernel size is kernsizeXkernsize
struct Tensor *maxpool(struct Tensor *x, int kernsize, int pads, int stride)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // the shape of the result tensor is:
    struct Tensor *result = getTensor(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = x->lens[1];
    result->lens[2] = (x->lens[2] + pads * 2 - kernsize) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - kernsize) / stride + 1;
    result->data = (DTYPE *)malloc(sizeof(DTYPE) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);
    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < x->lens[1]; i++)
    { // for each feature map,indexed by i
      // printf("i=%d\n",i);
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++)
            {
                float max=-10000;
                for (int m = 0; m < kernsize; m++)
                {
                    for (int n = 0; n < kernsize; n++)
                    { // for each kernel point,indexed by (m,n)
                        DTYPE xdat = (j * stride + m  < pads || j * stride + m  >= pads + x->lens[2] || k * stride + n  < pads || k * stride + n  >= pads + x->lens[3]) ? 0 : x->data[getindex(4, 0, i, j * stride + m - pads, k * stride + n - pads, x->lens)];
                        //printf("xdat=%f\n",xdat);
                        if(xdat>max) max=xdat;
                    }
                }
                //printf("max=%f\n",max);
                result->data[getindex(4, 0, i, j, k, result->lens)] = max;
            }
        }
    }

    return result;
}

struct TensorQ *Qmaxpool(struct TensorQ *x, int kernsize, int pads, int stride)
{
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    // the shape of the result tensor is:
    struct TensorQ *result = getTensorQ(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = x->lens[1];
    result->lens[2] = (x->lens[2] + pads * 2 - kernsize) / stride + 1;
    result->lens[3] = (x->lens[3] + pads * 2 - kernsize) / stride + 1;
    result->data = (int *)malloc(sizeof(int) * result->lens[0] * result->lens[1] * result->lens[2] * result->lens[3]);
    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < x->lens[1]; i++)
    { // for each feature map,indexed by i
      // printf("i=%d\n",i);
        for (int j = 0; j < result->lens[2]; j++)
        {
            for (int k = 0; k < result->lens[3]; k++)
            {
                int max=-10000;
                for (int m = 0; m < kernsize; m++)
                {
                    for (int n = 0; n < kernsize; n++)
                    { // for each kernel point,indexed by (m,n)
                        int xdat = (j * stride + m  < pads || j * stride + m  >= pads + x->lens[2] || k * stride + n  < pads || k * stride + n  >= pads + x->lens[3]) ? -10000 : x->data[getindex(4, 0, i, j * stride + m - pads, k * stride + n - pads, x->lens)];
                        //if((xdat>=128 || xdat<=-129)&& xdat!=-10000) printf("xdat=%f\n",xdat);
                        if(xdat>max) max=xdat;
                    }
                }
                //printf("max=%f\n",max);
                result->data[getindex(4, 0, i, j, k, result->lens)] = max;
            }
        }
    }

    return result;
}