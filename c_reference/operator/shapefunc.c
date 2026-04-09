// concat,split,...
#include "../tensor.h"
#include "../basic.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int checkshape_axi(struct Tensor *x, struct Tensor *y, int axi)
{
    if (x->ndim != y->ndim)
        return 0;
    if (axi >= x->ndim)
        return 0;
    for (int i = 0; i < x->ndim; i++)
    {
        if (i != axi && x->lens[i] != y->lens[i])
            return 0;
    }
    return 1;
}

struct Tensor *concat(struct Tensor *x, struct Tensor *y, int axi)
{
    if (!checkshape_axi(x, y, axi))
        return 0;
    int lenth = getlength(x) + getlength(y);
    struct Tensor *ret = getTensor(x->ndim);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * lenth);
    for (int i = 0; i < x->ndim; i++)
    {
        if (i == axi)
            ret->lens[i] = x->lens[i] + y->lens[i];
        else
            ret->lens[i] = x->lens[i];
    }
    int xaxilen = getaxilen(x, axi);
    int yaxilen = getaxilen(y, axi);
    for (int index = 0, indexx = 0, indexy = 0; index < lenth;)
    {
        memcpy(ret->data + index, x->data + indexx, sizeof(DTYPE) * xaxilen);
        indexx += xaxilen;
        index += xaxilen;
        memcpy(ret->data + index, y->data + indexy, sizeof(DTYPE) * yaxilen);
        indexy += yaxilen;
        index += yaxilen;
    }
    return ret;
}

struct TensorQ *Qconcat(struct TensorQ *x1, struct TensorQ *y1, int axi,int zx,int zw,int zo,long long factor,int pown,long long factor2,int pown2)
{
    //if (!checkshape_axi(x, y, axi))
        //return 0;
    struct TensorQ *x=copyTensorQ(x1);
    struct TensorQ *y=copyTensorQ(y1);
    int lenth = getlengthQ(x) + getlengthQ(y);
    struct TensorQ *ret = getTensorQ(x->ndim);
    ret->data = (int *)malloc(sizeof(int) * lenth);
    //modify x and y to output quantization
    int lenx=getlengthQ(x);
    int leny=getlengthQ(y);
    for(int i=0;i<lenx;i++){
        // x->data[i]=result_product_gen_32bit(x->data[i],factor,zo);
        x->data[i]=factor_product(x->data[i]-zx,factor,pown)+zo;
    }
    for(int i=0;i<leny;i++){
        // y->data[i]=result_product_gen_32bit(y->data[i],factor2,zo)
        y->data[i]=factor_product(y->data[i]-zw,factor2,pown2)+zo;
    }
    for (int i = 0; i < x->ndim; i++)
    {
        if (i == axi)
            ret->lens[i] = x->lens[i] + y->lens[i];
        else
            ret->lens[i] = x->lens[i];
    }
    int xaxilen = getaxilenQ(x, axi);
    int yaxilen = getaxilenQ(y, axi);
    for (int index = 0, indexx = 0, indexy = 0; index < lenth;)
    {
        memcpy(ret->data + index, x->data + indexx, sizeof(int) * xaxilen);
        indexx += xaxilen;
        index += xaxilen;
        memcpy(ret->data + index, y->data + indexy, sizeof(int) * yaxilen);
        indexy += yaxilen;
        index += yaxilen;
    }
    freeTensorQ(x);freeTensorQ(y);
    return ret;
}

struct Tensor *split(struct Tensor *x, int axi, int i, int len)
{
    if (axi >= x->ndim)
        return 0;
    if (x->lens[axi] < i + len)
        return 0;
    struct Tensor *ret = getTensor(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        if (i != axi)
            ret->lens[i] = x->lens[i];
        else
            ret->lens[i] = len;
    }
    int length = getlength(ret);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * length);
    int length0 = getlength(x);
    int axilen = getaxilen(x, axi);
    int axi1len = getaxilen(x, axi + 1);
    int indexsrc = axi1len * i;
    int indexdst = 0;
    //printf("axilen=%d and exi1len=%d\n",axilen,axi1len);
    while (indexsrc < length0)
    {
        memcpy(ret->data + indexdst, x->data + indexsrc, len * axi1len * sizeof(DTYPE));
        indexdst += len * axi1len;
        indexsrc += axilen;
    }
    return ret;
}

struct Tensor *reshape(struct Tensor *x, int ndims, ...)
{
    va_list ap;
    va_start(ap, ndims);
    int len = 1;
    for (int i = 0; i < ndims; i++)
    {
        len *= va_arg(ap, int);
    }
    if (len != getlength(x))
        return 0;
    struct Tensor *ret = getTensor(ndims);
    va_start(ap, ndims);
    for (int i = 0; i < ndims; i++)
    {
        ret->lens[i] = va_arg(ap, int);
    }
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * len);
    memcpy(ret->data, x->data, sizeof(DTYPE) * len);
    return ret;
}

// input the permutations
// output the tranposed matrix
struct Tensor *transpose(struct Tensor *x, ...)
{

    int ndim = x->ndim;
    struct Tensor *ret = copyTensor(x);
    va_list ap;
    va_start(ap, x);
    int *dest = (int *)malloc(sizeof(int) * x->ndim);
    int *current = (int *)malloc(sizeof(int) * x->ndim);
    for (int i = 0; i < ndim; i++)
    {
        current[i] = i;
        dest[i] = va_arg(ap, int);
    }
    int steps = 1;
    int len = getlength(x);
    DTYPE *data = (DTYPE *)malloc(sizeof(DTYPE) * len);
    for (int i = 0; i < ndim - 1; i++)
    {
        int repeat = x->lens[dest[i]];
        int cplen = 1;
        int j;
        int p;
        for (j = 0; j < ndim; j++)
        {
            if (current[j] == dest[i])
            {
                p = j;
                break;
            }
        }
        int t = current[p];
        j++;
        for (; j < ndim; j++)
        {
            cplen *= x->lens[current[j]];
        }
        int stride = cplen * repeat;
        // rearrange axis
        for (j = p; j >= i + 1; j--)
        {
            current[j] = current[j - 1];
        }
        current[i] = t;
        int stepoffset = len / steps;
        // printf("steps=%d,repeat=%d cplen=%d stride=%d\n",steps,repeat,cplen,stride);
        for (int s = 0; s < steps; s++)
        {
            int cpdst = stepoffset * s;
            int cpsrc0 = stepoffset * s;
            int cpsrcend = cpsrc0 + stepoffset;
            // printf("cpdst=%d,cpsrc0=%d,cpsrcend=%d\n", cpdst, cpsrc0, cpsrcend);
            for (int j = 0; j < repeat; j++)
            {
                // printf("repeat=%d\n", j);
                for (int cpsrc = cpsrc0 + j * cplen; cpsrc < cpsrcend; cpsrc += stride, cpdst += cplen)
                {
                    memcpy(data + cpdst, ret->data + cpsrc, cplen * sizeof(DTYPE));
                }
            }
        }
        memcpy(ret->data, data, len * sizeof(DTYPE));

        // exit(0);
        steps *= repeat;
    }
    free(data);
    free(current);
    for (int i = 0; i < ndim; i++)
    {
        ret->lens[i] = x->lens[dest[i]];
    }
    free(dest);
    return ret;
}

// output the tranposed matrix
struct TensorQ *transposeQ(struct TensorQ *x, ...)
{

    int ndim = x->ndim;
    struct TensorQ *ret = copyTensorQ(x);
    va_list ap;
    va_start(ap, x);
    int *dest = (int *)malloc(sizeof(int) * x->ndim);
    int *current = (int *)malloc(sizeof(int) * x->ndim);
    for (int i = 0; i < ndim; i++)
    {
        current[i] = i;
        dest[i] = va_arg(ap, int);
    }
    int steps = 1;
    int len = getlengthQ(x);
    int *data = (int *)malloc(sizeof(int) * len);
    for (int i = 0; i < ndim - 1; i++)
    {
        int repeat = x->lens[dest[i]];
        int cplen = 1;
        int j;
        int p;
        for (j = 0; j < ndim; j++)
        {
            if (current[j] == dest[i])
            {
                p = j;
                break;
            }
        }
        int t = current[p];
        j++;
        for (; j < ndim; j++)
        {
            cplen *= x->lens[current[j]];
        }
        int stride = cplen * repeat;
        // rearrange axis
        for (j = p; j >= i + 1; j--)
        {
            current[j] = current[j - 1];
        }
        current[i] = t;
        int stepoffset = len / steps;
        // printf("steps=%d,repeat=%d cplen=%d stride=%d\n",steps,repeat,cplen,stride);
        for (int s = 0; s < steps; s++)
        {
            int cpdst = stepoffset * s;
            int cpsrc0 = stepoffset * s;
            int cpsrcend = cpsrc0 + stepoffset;
            // printf("cpdst=%d,cpsrc0=%d,cpsrcend=%d\n", cpdst, cpsrc0, cpsrcend);
            for (int j = 0; j < repeat; j++)
            {
                // printf("repeat=%d\n", j);
                for (int cpsrc = cpsrc0 + j * cplen; cpsrc < cpsrcend; cpsrc += stride, cpdst += cplen)
                {
                    memcpy(data + cpdst, ret->data + cpsrc, cplen * sizeof(int));
                }
            }
        }
        memcpy(ret->data, data, len * sizeof(int));

        // exit(0);
        steps *= repeat;
    }
    free(data);
    free(current);
    for (int i = 0; i < ndim; i++)
    {
        ret->lens[i] = x->lens[dest[i]];
    }
    free(dest);
    return ret;
}