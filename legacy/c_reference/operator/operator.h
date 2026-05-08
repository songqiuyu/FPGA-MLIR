#include "../tensor.h"
#include <stdlib.h>
#pragma once

struct Tensor * conv(struct Tensor *x,struct Tensor *w,struct Tensor *b,int dilation,int pads,int stride);
struct Tensor * conv2(struct Tensor *x, struct Tensor *w, struct Tensor *b, int dilation, int pads, int stride);
struct TensorQ *conv2Q(struct TensorQ *x, struct TensorQ *w, struct TensorQ *b, int dilation, int pads,int stride,int zx,int zw,int zo,long long factor,int pown);
struct TensorQ *depthwiseconv2Q(struct TensorQ *x, struct TensorQ *w, struct TensorQ *b, int dilation, int pads, int stride, int zx, int zw, int zo, long long factor, int pown);
struct Tensor *maxpool(struct Tensor *x, int k, int pads, int stride);
struct TensorQ *Qmaxpool(struct TensorQ *x, int kernsize, int pads, int stride);

struct Tensor *sigmoid(struct Tensor *x);
struct TensorQ *Qsigmoid(struct TensorQ *x,double sx,int zx,double sy,int zy);
struct Tensor *mul(struct Tensor *x, struct Tensor *y);
struct TensorQ *Qmul(struct TensorQ *x, struct TensorQ *y,int zx,int zw,int zo,long long factor,int pown);
struct Tensor *add(struct Tensor *x, struct Tensor *y);
struct TensorQ *Qadd(struct TensorQ *x, struct TensorQ *y,int zx,int zw,int zo,long long factor,int pown,long long factor2,int pown2);
struct Tensor *power(struct Tensor *x, struct Tensor *y);

struct Tensor *concat(struct Tensor *x, struct Tensor *y, int axi);
struct TensorQ *Qconcat(struct TensorQ *x1, struct TensorQ *y1, int axi,int zx,int zw,int zo,long long factor,int pown,long long factor2,int pown2);
struct Tensor * reshape(struct Tensor *x, int ndims, ...);
struct Tensor *transpose(struct Tensor *x,...);
struct TensorQ *transposeQ(struct TensorQ *x, ...);
struct Tensor *split(struct Tensor *x, int axi, int i, int len);

struct Tensor *resize_nni(struct Tensor *x, int n, int m, int mode);
struct TensorQ *Qresize_nni(struct TensorQ *x, int n, int m, int mode);

int bias_gen(const char *name,int m) //32-bit bias value of each layer is stored in 'name' file
{
    static char buf[1000];
    #ifdef BIAS_GEN
        int m16=((m+15)>>4);
        sprintf(buf,"0x%x:%s\n",bias_addr,name);
        FILE *fp=fopen(name,"rb");
        fwrite(buf,1,strlen(buf),fp_bias_addr);
        fseek(fp,0,SEEK_END);
        int size=ftell(fp);
        fseek(fp,0,SEEK_SET);
        static int buf16[16];
        int sm=0;
        while(fread(buf16,__min(m-sm,16)*4,1,fp)){
            for(int k=16-1;k>=0;k--)
            {
                fprintf(fp_bias_coe,"%08x",buf16[k]);
            }
            fwrite(buf16,1,4*16,fp_bias_image);
            sm=sm+16;
            fprintf(fp_bias_coe,",");
        }
        bias_addr+=m16;
    #endif
}

int weight_gen(struct TensorQ *w,const char *name)
{
    static char buf[1000];
    #ifdef WEIGHT_GEN
        int n=w->lens[1];
        int n32=((n+31)>>5)<<5;
        int kernel=w->lens[2];
        sprintf(buf,"0x%x:%s\n",weight_addr,name);
        fwrite(buf,1,strlen(buf),fp_weight_addr);
        int m=w->lens[0];
        struct TensorQ *wt=transposeQ(w,0,2,3,1);
        int tlen=getlengthQ(wt);
        int wt_data_offset=0;
        int rem=tlen;
        static char buf32[32];
        int sn=0;
        while(rem){
            int cplen=__min(32,n-sn);
            for(int k=0;k<cplen;k++)
            {
                buf32[k]=wt->data[wt_data_offset++];
            }
            fwrite(buf32,32,1,fp_weight_image);
            sn=sn+32>=n? 0:sn+32;
            rem-=cplen;
        }
        weight_addr+=n32*m*kernel*kernel;
        freeTensorQ(wt);
    #endif
}



#define Conv(name,x1,x2,x3,x4,d,p,s,nw,nb,input,output,del) \
w=getTensor(4); \
w->lens[0]=x1;w->lens[1]=x2;w->lens[2]=x3;w->lens[3]=x4;  \
w->data=(DTYPE*)readfile(nw); \
b=getTensor(1);\
b->lens[0]=x1;\
b->data=(DTYPE*)readfile(nb);\
struct Tensor *output=conv2(input,w,b,d,p,s);\
freeTensor(w);freeTensor(b);\
if(del)\
freeTensor(input);\
printf("%s ok\n",name);

#define QLinearConv(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,dest_addr,name,input,nsx,nzx,nw,nsw,nzw,nso,nzo,nb,output,d,k,p,s) \
wq=getTensorQ(4); \
printf("=========================\n");\
printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n",nsx,nzx,nsw,nzw,nso,nzo,nw,nb);\
printf("=========================\n");\
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(float)(*sx)*(float)(*sw)/(float)(*so);\
printf("conv %s.factor=%d\n",name,getfactor(factor,36));\
wq->lens[0]=filesize(nw)/input->lens[1]/k/k;wq->lens[1]=input->lens[1];wq->lens[2]=k;wq->lens[3]=k;  \
tlen=getlengthQ(wq);\
tdataint8=(char*)readfile(nw); \
wq->data=(int *)malloc(sizeof(int)*tlen);\
for(int i=0;i<tlen;i++){\
    wq->data[i]=tdataint8[i];\
}\
free(tdataint8);\
bq=getTensorQ(1);\
bq->lens[0]=wq->lens[0];\
tlen=getlengthQ(bq);\
bq->data=(int *)readfile(nb);\
struct TensorQ *output=conv2Q(input,wq,bq,d,p,s,*zx,*zw,*zo,getfactor(factor,36),36);\
conv_instruction_gen(fpo,fp_hex,input,wq,d,wq->lens[2],s,p,tM,tR,tC,sM,Mconcat,*zx,*zw,*zo,source_addr,dest_addr,getfactor(factor,36));\
bias_gen(nb,wq->lens[0]);\
weight_gen(wq,nw);\
freeTensorQ(wq);freeTensorQ(bq);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);\
printshapeQ(output);\

#define QLinearConv_AUTO(fpo,fp_hex,source_addr,dest_addr,name,input,nsx,nzx,nw,nsw,nzw,nso,nzo,nb,output,d,k,p,s) \
wq=getTensorQ(4); \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
printf("=========================\n");\
printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n",nsx,nzx,nsw,nzw,nso,nzo,nw,nb);\
printf("=========================\n");\
factor=(float)(*sx)/(float)(*so)*(float)(*sw);\
printf("conv %s.factor=%d\n",name,getfactor(factor,36));\
wq->lens[0]=filesize(nw)/input->lens[1]/k/k;wq->lens[1]=input->lens[1];wq->lens[2]=k;wq->lens[3]=k;  \
tlen=getlengthQ(wq);\
tdataint8=(char*)readfile(nw); \
wq->data=(int *)malloc(sizeof(int)*tlen);\
for(int i=0;i<tlen;i++){\
    wq->data[i]=tdataint8[i];\
}\
free(tdataint8);\
bq=getTensorQ(1);\
bq->lens[0]=wq->lens[0];\
tlen=getlengthQ(bq);\
bq->data=(int *)readfile(nb);\
struct TensorQ *output=conv2Q(input,wq,bq,d,p,s,*zx,*zw,*zo,getfactor(factor,36),36);\
getslice(input,wq,p,d,s,tX); \
conv_instruction_gen(fpo,fp_hex,input,wq,d,wq->lens[2],s,p,tX[0],tX[1],tX[2],0,wq->lens[0],*zx,*zw,*zo,source_addr,dest_addr,getfactor(factor,36));\
bias_gen(nb,wq->lens[0]);\
weight_gen(wq,nw);\
freeTensorQ(wq);freeTensorQ(bq);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);\
printshapeQ(output);\
name_param_idx+=1;\
quantized_param_idx+=2;

#define QLinearDepthConv(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,dest_addr,name,input,nsx,nzx,nw,nsw,nzw,nso,nzo,nb,output,d,k,p,s)  \
wq=getTensorQ(4); \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo); \
factor=(float)(*sx)*(float)(*sw)/(float)(*so);\
wq->lens[0]=filesize(nw)/k/k;wq->lens[1]=1;wq->lens[2]=k;wq->lens[3]=k;\
tlen=getlengthQ(wq);\
tdataint8=(char*)readfile(nw);\
wq->data=(int *)malloc(sizeof(int)*tlen);\
for(int i=0;i<tlen;i++){\
    wq->data[i]=tdataint8[i];\
}\
free(tdataint8);\
bq=getTensorQ(1);\
bq->lens[0]=wq->lens[0];\
tlen=getlengthQ(bq);\
bq->data=(int *)readfile(nb);\
struct TensorQ *output=depthwiseconv2Q(input,wq,bq,d,p,s,*zx,*zw,*zo,getfactor(factor,36),36);\
depthconv_instruction_gen(fpo,fp_hex,input,wq,d,wq->lens[2],s,p,tM,tR,tC,sM,Mconcat,*zx,*zw,*zo,source_addr,dest_addr,getfactor(factor,36));\
bias_gen(nb,wq->lens[0]);\
weight_gen(wq,nw);\
silu_gen_eq(name);\
freeTensorQ(wq);freeTensorQ(bq);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);\
printshapeQ(output);

#define QLinearDepthConvWithSiLU(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,dest_addr,name,input,nsx,nzx,nw,nsw,nzw,nso,nzo,nb,output,d,k,p,s)  \
wq=getTensorQ(4); \
printf("=========================\n");\
printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n",nsx,nzx,nsw,nzw,nso,nzo,nw,nb);\
printf("=========================\n");\
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo); \
factor=(float)(*sx)*(float)(*sw)/(float)(*so);\
wq->lens[0]=filesize(nw)/k/k;wq->lens[1]=1;wq->lens[2]=k;wq->lens[3]=k;\
tlen=getlengthQ(wq);\
tdataint8=(char*)readfile(nw);\
wq->data=(int *)malloc(sizeof(int)*tlen);\
for(int i=0;i<tlen;i++){\
    wq->data[i]=tdataint8[i];\
}\
free(tdataint8);\
bq=getTensorQ(1);\
bq->lens[0]=wq->lens[0];\
tlen=getlengthQ(bq);\
bq->data=(int *)readfile(nb);\
struct TensorQ *output=depthwiseconv2Q(input,wq,bq,d,p,s,*zx,*zw,*zo,getfactor(factor,36),36);\
depthconv_instruction_gen(fpo,fp_hex,input,wq,d,wq->lens[2],s,p,tM,tR,tC,sM,Mconcat,*zx,*zw,*zo,source_addr,dest_addr,getfactor(factor,36));\
bias_gen(nb,wq->lens[0]);\
weight_gen(wq,nw);\
freeTensorQ(wq);freeTensorQ(bq);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);\
printshapeQ(output);


#define Split(name,input,output,del,axi,i,len) \
struct Tensor *output=split(input,axi,i,len);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define Transpose(name,input,output,del,...) \
struct Tensor *output=transpose(input,__VA_ARGS__);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define Reshape(name,input,output,ndim,del,...) \
struct Tensor *output=reshape(input,ndim,__VA_ARGS__);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define Resize_nni(name,input,output,n,m,mode,del) \
struct Tensor *output=resize_nni(input,n,m,mode);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define QResize(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,dest_addr,name,input,dummy,nscale,output) \
sx=(float *)readfile(nscale);\
n=round(input->lens[2]*sx[2]);\
m=round(input->lens[3]*sx[3]);\
struct TensorQ *output=Qresize_nni(input,n,m,0);\
usample_instruction_gen(fpo,fp_hex,input,tM,tR,tC,sM,Mconcat,source_addr,dest_addr);\
printf("%s ok\n",name); \
printshapeQ(output);\
free(sx);

#define Maxpool(name,input,output,k,p,s,del) \
struct Tensor *output=maxpool(input,k,p,s);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define QMaxPool(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,dest_addr,name,input,output,k,p,s) \
struct TensorQ *output=Qmaxpool(input,k,p,s);\
mpool_instruction_gen(fpo,fp_hex,input,tM,tR,tC,p,k,s,sM,Mconcat,source_addr,dest_addr);\
printf("%s ok\n",name); \
printshapeQ(output);

#define Sigmoid(name,input,output,del) \
struct Tensor *output=sigmoid(input);\
printf("%s ok\n",name); \
if(del) freeTensor(input);

#define QLinearSiLU_AUTO(name,name2,input,nsx,nzx,nsw,nzw,nso,nzo,output) \
sx=(float *)readfile(nsx);so=(float *)readfile(nsw);\
zx=(long long *)readfile(nzx);zo=(long long *)readfile(nzw);\
printf("=========================\n");\
printf("%s\n%s\n%s\n%s\n%s\n%s\n",nsx,nzx,nsw,nzw,nso,nzo);\
printf("=========================\n");\
input_sig=Qsigmoid(input,*sx,*zx,*so,*zo);\
free(sx);free(so);free(zx);free(zo);\
cliptensorQ(input_sig);\
printf("%s ok\n",name);\
name_param_idx+=1; \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(double)(*sx)*(*sw)/(*so);\
struct TensorQ *output=Qmul(input,input_sig,*zx,*zw,*zo,getfactor(factor,28),28);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
silu_gen(name2,nsx,nzx,nsw,nzw,nso,nzo);\
printf("%s ok\n",name2); \
name_param_idx+=1; \
scale_param_idx+=4;\
zero_point_param_idx+=4;

#define QLinearSiLU_AUTO_SPLIT(name,name2,input,nsx,nzx,nsw,nzw,nso,nzo,output) \
sx=(float *)readfile(nsx);so=(float *)readfile(nsw);\
zx=(long long *)readfile(nzx);zo=(long long *)readfile(nzw);\
input_sig=Qsigmoid(input,*sx,*zx,*so,*zo);\
free(sx);free(so);free(zx);free(zo);\
cliptensorQ(input_sig);\
printf("%s ok\n",name);\
name_param_idx+=1; \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(double)(*sx)*(*sw)/(*so);\
struct TensorQ *output=Qmul(input,input_sig,*zx,*zw,*zo,getfactor(factor,28),28);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
silu_gen(name2,nsx,nzx,nsw,nzw,nso,nzo);\
printf("%s ok\n",name2); \
name_param_idx+=1; \
scale_param_idx+=5;\
zero_point_param_idx+=5;


#define QLinearSigmoid(name,input,nsx,nzx,nso,nzo,output) \
sx=(float *)readfile(nsx);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zo=(long long *)readfile(nzo);\
struct TensorQ *output=Qsigmoid(input,*sx,*zx,*so,*zo);\
free(sx);free(so);free(zx);free(zo);\
cliptensorQ(output);

#define Mul(name,input1,input2,output,del) \
struct Tensor *output=mul(input1,input2);\
printf("%s ok\n",name);\
if(del){\
freeTensor(input1);freeTensor(input2);\
}

#define QLinearMul(name,input1,nsx,nzx,input2,nsw,nzw,nso,nzo,output) \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(double)(*sx)*(*sw)/(*so);\
struct TensorQ *output=Qmul(input1,input2,*zx,*zw,*zo,getfactor(factor,28),28);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
silu_gen(name,nsx,nzx,nsw,nzw,nso,nzo);

#define QLinearMul_Nonrec(name,input1,nsx,nzx,input2,nsw,nzw,nso,nzo,output) \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(double)(*sx)*(*sw)/(*so);\
struct TensorQ *output=Qmul(input1,input2,*zx,*zw,*zo,getfactor(factor,28),28);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);

#define Mul_cont(name,input,output,cont,del) \
t=copyTensor(input);\
tlen=getlength(t);\
for(int i=0;i<tlen;i++){\
    t->data[i]=cont;\
}\
struct Tensor *output=mul(input,t);\
printf("%s ok\n",name);\
freeTensor(t);\
if(del){\
freeTensor(input);\
}

#define Pow_cont(name,input,output,cont,del) \
t=copyTensor(input);\
tlen=getlength(t);\
for(int i=0;i<tlen;i++){\
    t->data[i]=cont;\
}\
struct Tensor *output=power(input,t);\
printf("%s ok\n",name);\
freeTensor(t);\
if(del){\
freeTensor(input);\
}

#define Add(name,input1,input2,output,del) \
struct Tensor *output=add(input1,input2);\
if(del){\
freeTensor(input1);freeTensor(input2);\
}\
printf("%s ok\n",name);\
tlen=getlength(output);\
for(int i=0;i<tlen;i++){\
    if(output->data[i]>1 || output->data[i]<-1){\
        printf("fatal %f\n",output->data[i]);\
    }\
}

#define QLinearAdd(fpo,fp_hex,tM,tR,tC,Mconcat,sM,source_addr,source_addr2,dest_addr,name,input1,nsx,nzx,input2,nsw,nzw,nso,nzo,output) \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(*sx)/(double)(*so);\
factor2=(*sw)/(double)(*so);\
struct TensorQ *output=Qadd(input1,input2,*zx,*zw,*zo,getfactor(factor,28),28,getfactor(factor2,28),28);\
res_instruction_gen(fpo,fp_hex,input1,input2,tM,tR,tC,sM,Mconcat,*zo,*zx,*zw,source_addr,source_addr2,dest_addr,getfactor(factor,28),getfactor(factor2,28));\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);

#define Add_cont_tensor(name,input,input2name,output,del) \
t=copyTensor(input);\
free(t->data);\
t->data=(DTYPE*)readfile(input2name);\
struct Tensor *output=add(input,t);\
freeTensor(t);\
if(del){\
freeTensor(input);\
}\
printf("%s ok\n",name);

#define Mul_cont_tensor(name,input,input2name,output,del) \
t=copyTensor(input);\
free(t->data);\
t->data=(DTYPE*)readfile(input2name);\
struct Tensor *output=mul(input,t);\
freeTensor(t);\
if(del){\
freeTensor(input);\
}\
printf("%s ok\n",name);

#define Concat(name,input1,input2,output,axi,del) \
struct Tensor *output=concat(input1,input2,axi);\
if(del){\
freeTensor(input1);freeTensor(input2);\
}\
printf("%s ok\n",name);

#define QLinearConcat(fpo,fp_hex,tM,tR,tC,halfM,Mconcat,source_addr,source_addr2,dest_addr,name,nso,nzo,input1,nsx,nzx,input2,nsw,nzw,output,axi) \
sx=(float *)readfile(nsx);sw=(float *)readfile(nsw);so=(float *)readfile(nso);\
zx=(long long *)readfile(nzx);zw=(long long *)readfile(nzw);zo=(long long *)readfile(nzo);\
factor=(*sx)/(double)(*so);\
factor2=(*sw)/(double)(*so);\
struct TensorQ *output=Qconcat(input1,input2,axi,*zx,*zw,*zo,getfactor(factor,28),28,getfactor(factor2,28),28);\
res_instruction_gen(fpo,fp_hex,input1,input1,tM,tR,tC,0,Mconcat,*zo,*zx,123,source_addr,source_addr,dest_addr,getfactor(factor,28),0);\
res_instruction_gen(fpo,fp_hex,input2,input2,tM,tR,tC,halfM,Mconcat,*zo,*zw,123,source_addr2,source_addr2,dest_addr,getfactor(factor2,28),0);\
free(sw);free(sx);free(so);free(zx);free(zw);free(zo);\
cliptensorQ(output);\
printf("%s ok\n",name);

#define DequantizeLinear(name,input1,in1_s,in1_z,output)\
sx=(double *)readfile(in1_s);\
zx=(long long *)readfile(in1_z);\
struct Tensor *output=convertTensor(input1, *sx, *zx);\
free(sx);free(zx);freeTensor(input1);\
printf("%s ok\n",name);
