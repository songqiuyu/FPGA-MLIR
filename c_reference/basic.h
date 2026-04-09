#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tensor.h"
#include <stdint.h>
int getindex(int ndims,...);

struct Tensor *getTensor(int ndim);
struct TensorQ *getTensorQ(int ndim);

void freeTensor(struct Tensor *tensor);
void freeTensorQ(struct TensorQ *tensor);

void printshape(struct Tensor *t);
void printshapeQ(struct TensorQ *t);

unsigned char *readfile(const char *filename);
int filesize(const char *filename);

int getlength(struct Tensor *x);
int getlengthQ(struct TensorQ *x);

int getaxilen(struct Tensor *x, int axi);
int getaxilenQ(struct TensorQ *x, int axi);

void compareTensor(struct Tensor *x, struct Tensor *y);

struct Tensor *copyTensor(struct Tensor *x);
struct TensorQ *copyTensorQ(struct TensorQ *x);
struct Tensor *convertTensor(struct TensorQ *x, double s, int z);

long long getfactor(double f,int bits);

int getpown(double f);
int factor_product(int x, long long factor, int pown);
int factor_product_add(int x, long long factor, int pown, int y, long long factor2, int pown2);
int result_product_gen_32bit(int x, long long factor, int zo);
int result_product_add_gen_32bit(int x, long long factor, int y, long long factor2, int zx, int zw, int zo);


void cliptensorQ(struct TensorQ *x);

int getslice(struct TensorQ *x, struct TensorQ *w, int pad, int d, int s, int tX[3]);
int calculate_buffer_consumption(int tN, int tM, int tR, int tC, int N, int M, int kernel, int stride, int pad, int dilation);

int test_ff(FILE *fpo, long long address, long long length);
int transmit_instruction_gen(FILE *fpo, int address, int length);
int conv_instruction_gen(FILE *fpo,struct TensorQ *x,struct TensorQ *w,int d,int k,int s,int pad,int tM,int tR,int tC,int sM,int Mconcat,int zx,int zw,int zo,unsigned long long source_addr,unsigned long long dest_addr,unsigned long long factor);
int res_instruction_gen(FILE *fpo,struct TensorQ *x,struct TensorQ *w,int tM,int tR,int tC,int sM,int Mconcat,int yz,int xz,int wz,unsigned long long source_addr,unsigned long long source_addr2,unsigned long long dest_addr,long long factor,long long factor2);
int mpool_instruction_gen(FILE *fpo,struct TensorQ *x,int tM,int tR,int tC,int pad,int kernel,int stride,int sM,int Mconcat,unsigned long long source_addr,unsigned long long dest_addr);
int usample_instruction_gen(FILE *fpo,struct TensorQ *x,int tM,int tR,int tC,int sM,int Mconcat,unsigned long long source_addr,unsigned long long dest_addr);
int depthconv_instruction_gen(FILE *fpo, struct TensorQ *x, struct TensorQ *w, int d, int k, int s, int pad, int tM, int tR, int tC, int sM, int Mconcat, int zx, int zw, int zo, unsigned long long source_addr, unsigned long long dest_addr, unsigned long long factor);


void processFractionalPart(uint32_t fractional_part, uint32_t *fractional_parts);
void convertToIEEE754(uint32_t sign, uint32_t integer_part, uint32_t fractional_part, uint32_t *result_final);
int my_nearbyint(uint32_t x);
uint32_t ieee754_float_add(uint32_t sign1, uint32_t exponent1, uint32_t significant1,
                           uint32_t sign2, uint32_t exponent2, uint32_t significant2);
int result_compare(int x0, int x1, int x2, int x3, struct TensorQ *x, const char *filename);


//VLIW structure
struct VLIW{
    long long operator;
    long long DDR_x1_address;
    long long DDR_x2_address;
    long long Bias_source_address;
    long long Compute_Result_dest_address;
    long long Activate_LUT_address;
    long long R;
    long long C;
    long long M;
    long long N;
    long long R0;
    long long C0;
    long long sM_concat;
    long long M_concat;
    long long Quant_x1_z;
    long long Quant_x2_z;
    long long Quant_y_z;
    long long Conv_pad;
    long long Conv_kernel;
    long long Conv_stride;
    long long Conv_dilation;
    long long Conv_tR;
    long long Conv_tC;
    long long Conv_tM;
    long long Conv_tN;
    long long Conv_permuteR;
    long long Conv_permuteC;
    long long Conv_permuteM;
    long long Conv_permuteN;
    long long Conv_quant_factor;
    long long Conv_quant_factor2;
};