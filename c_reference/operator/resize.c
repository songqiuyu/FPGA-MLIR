// commonly referred to as 'interpolation',or 'upsampling'
#include "../tensor.h"
#include "../basic.h"
#include <math.h>
#include <stdio.h>
#include <omp.h>



// nearest neighbor interpolation
// mode:0:floor
// mode:1:round prefer up
// mode:2:round prefer down
struct Tensor *resize_nni(struct Tensor *x, int n, int m, int mode)
{
    double (*modefunc) (double );
    if(mode==0) modefunc=floor;
    if(mode==1) modefunc=round;
    if(mode==1) modefunc=trunc;
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    double scalen = (double)n / x->lens[2];
    double scalem = (double)m / x->lens[3];

    struct Tensor *result = getTensor(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = x->lens[1];
    result->lens[2] = n;
    result->lens[3] = m;
    result->data=(DTYPE *)malloc(sizeof(DTYPE)*getlength(result));

    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < x->lens[1]; i++)
    { // for each feature map,indexed by i
        for(int j=0;j<result->lens[2];j++){
            for(int k=0;k<result->lens[3];k++){
                //for each output point,indexed by (j,k)
                int j0=(int)modefunc(j/scalen);
                int k0=(int)modefunc(k/scalem);
                result->data[getindex(4,0,i,j,k,result->lens)]=x->data[getindex(4,0,i,j0,k0,x->lens)];
            }
        }
    }
    return result;
}

// nearest neighbor interpolation
// mode:0:floor
// mode:1:round prefer up
// mode:2:round prefer down
struct TensorQ *Qresize_nni(struct TensorQ *x, int n, int m, int mode)
{
    double (*modefunc) (double );
    if(mode==0) modefunc=floor;
    if(mode==1) modefunc=round;
    if(mode==1) modefunc=trunc;
    // x should have 4 dims
    if (x->ndim != 4)
        return 0;
    double scalen = (double)n / x->lens[2];
    double scalem = (double)m / x->lens[3];
    printf("scale=%lf and %lf\n",scalen,scalem);

    struct TensorQ *result = getTensorQ(x->ndim);
    result->lens[0] = x->lens[0];
    result->lens[1] = x->lens[1];
    result->lens[2] = n;
    result->lens[3] = m;
    result->data=(int *)malloc(sizeof(int)*getlengthQ(result));

    // the result tensor is:
#pragma omp parallel for
    for (int i = 0; i < x->lens[1]; i++)
    { // for each feature map,indexed by i
        for(int j=0;j<result->lens[2];j++){
            for(int k=0;k<result->lens[3];k++){
                //for each output point,indexed by (j,k)
                int j0=(int)modefunc(j/scalen);
                int k0=(int)modefunc(k/scalem);
                result->data[getindex(4,0,i,j,k,result->lens)]=x->data[getindex(4,0,i,j0,k0,x->lens)];
            }
        }
    }
    return result;
}