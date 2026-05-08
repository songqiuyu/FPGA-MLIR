// element wise operators such as mul add sigmoid ...
#include "../basic.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>

int checkshape(struct Tensor *x, struct Tensor *y)
{
    if (x->ndim != y->ndim)
        return 0;
    for (int i = 0; i < x->ndim; i++)
    {
        if (x->lens[i] != y->lens[i])
            return 0;
    }
    return 1;
}

struct Tensor *sigmoid(struct Tensor *x)
{
    struct Tensor *ret = getTensor(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlength(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * length);
    for (int i = 0; i < length; i++)
    {
        ret->data[i] = 1.0 / (1 + exp(-x->data[i]));
    }
    return ret;
}

struct TensorQ *Qsigmoid(struct TensorQ *x, double sx, int zx, double sy, int zy)
{
    struct TensorQ *ret = getTensorQ(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlengthQ(x);
    ret->data = (int *)malloc(sizeof(int) * length);
    for (int i = 0; i < length; i++)
    {
        double rx = sx * (x->data[i] - zx);
        double ry = 1.0 / (1 + exp(-rx));
        // ret->data[i] = round(ry / sy) + zy;
        ret->data[i] = nearbyint(ry / sy) + zy;
        // printf("data=%d\n",ret->data[i]);
    }
    return ret;
}

struct Tensor *mul(struct Tensor *x, struct Tensor *y)
{
    if (!checkshape(x, y))
        return 0;
    struct Tensor *ret = getTensor(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlength(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * length);
    for (int i = 0; i < length; i++)
    {
        ret->data[i] = x->data[i] * y->data[i];
    }
    return ret;
}

struct TensorQ *Qmul(struct TensorQ *x, struct TensorQ *y, int zx, int zw, int zo, long long factor, int pown)
{
    // if (!checkshape(x, y))
    // return 0;
    struct TensorQ *ret = getTensorQ(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlengthQ(x);
    ret->data = (int *)malloc(sizeof(int) * length);
    for (int i = 0; i < length; i++)
    {
        // if(i==315){
        //     printf("here");
        // }
        ret->data[i] = (x->data[i] - zx) * (y->data[i] - zw);
        // ret->data[i] = result_product_gen_32bit(ret->data[i], factor, zo);
        ret->data[i] = factor_product(ret->data[i], factor, pown) + zo;
    }
    return ret;
}

struct Tensor *power(struct Tensor *x, struct Tensor *y)
{
    if (!checkshape(x, y))
        return 0;
    struct Tensor *ret = getTensor(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlength(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * length);
    for (int i = 0; i < length; i++)
    {
        ret->data[i] = pow(x->data[i], y->data[i]);
    }
    return ret;
}

struct Tensor *add(struct Tensor *x, struct Tensor *y)
{
    if (!checkshape(x, y))
        return 0;
    struct Tensor *ret = getTensor(x->ndim);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlength(x);
    ret->data = (DTYPE *)malloc(sizeof(DTYPE) * length);
    for (int i = 0; i < length; i++)
    {
        ret->data[i] = x->data[i] + y->data[i];
    }
    return ret;
}

struct TensorQ *Qadd(struct TensorQ *x, struct TensorQ *y, int zx, int zw, int zo, long long factor, int pown, long long factor2, int pown2)
{
    // if (!checkshape(x, y))
    // return 0;
    struct TensorQ *ret = getTensorQ(x->ndim);
    // printf("from res: %d,%d,%d,%d\n", x->lens[0], x->lens[1], x->lens[2], x->lens[3]);
    for (int i = 0; i < x->ndim; i++)
    {
        ret->lens[i] = x->lens[i];
    }
    int length = getlengthQ(x);
    ret->data = (int *)malloc(sizeof(int) * length);
    for (int i = 0; i < length; i++)
    {
        // ret->data[i] = factor_product_add(x->data[i]-zx,factor,pown,y->data[i]-zw,factor2,pown2)+zo+zo-zo;
        if(i == getindex(4,0,97,75,10,ret->lens)){
            printf(".");
        }

        ret->data[i] = result_product_add_gen_32bit(x->data[i], factor, y->data[i], factor2, zx, zw, zo);
    }
    return ret;
}
