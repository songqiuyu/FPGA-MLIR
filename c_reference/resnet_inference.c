// yolov5_gray_640.onnx model implemented in C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "basic.h"

// global data for bias generation
#define BIAS_GEN   //for bias generation
#define WEIGHT_GEN //for weight generation
#define INPUT_GEN //for input image generation
#define SILU_GEN // for activation generation



#ifdef BIAS_GEN
int bias_addr;
FILE *fp_bias_coe;
FILE *fp_bias_addr;
FILE *fp_bias_image;
#endif
#ifdef WEIGHT_GEN
int weight_addr;
FILE *fp_weight_image;
FILE *fp_weight_addr;
#endif
#ifdef INPUT_GEN
FILE *fp_input_image;
#endif
#ifdef SILU_GEN
FILE *fp_silu_addr;
FILE *fp_silu_coe;
FILE *fp_silu_image;
int silu_addr;
#endif
#include "operator/operator.h"

int silu_gen(const char *name, const char *input_s, const char *input_z, const char *sigmoid_s, const char *sigmoid_z, const char *output_s, const char *output_z)
{
#ifdef SILU_GEN
    struct Tensor *w;
    struct TensorQ *wq;
    struct Tensor *b;
    struct TensorQ *bq;
    struct Tensor *t;
    float *sx;
    float *sw;
    float *so;
    long long *zx;
    long long *zw;
    long long *zo;
    double factor;
    double factor2;
    char *tdataint8;
    int n;
    int m;
    static char buf[1000];
    struct TensorQ *lut_x = getTensorQ(1);
    lut_x->lens[0] = 256;
    lut_x->data = (int *)malloc(sizeof(int) * 256);
    for (int i = 0; i < 256; i++)
    {
        lut_x->data[i] = (char)i;
    }
    QLinearSigmoid("silu_gen_sigmoid", lut_x, input_s, input_z, sigmoid_s, sigmoid_z, temp);
    QLinearMul_Nonrec("silu_gen_mul", temp, sigmoid_s, sigmoid_z, lut_x, input_s, input_z, output_s, output_z, lut_y);
    sprintf(buf, "0x%x:%s\n", silu_addr, name);
    fwrite(buf, 1, strlen(buf), fp_silu_addr);
    silu_addr+=1;
    for (int i = 255; i >= 0; i--)
    {
        // if(i==253) printf("253(-3)->%d\n",(char)(lut_y->data[i]&0xff));
        fprintf(fp_silu_coe, "%02x", lut_y->data[i] & 0xff);
    }
    for (int i = 0; i <256 ; i++)
    {
        fwrite(lut_y->data+i,1,1,fp_silu_image);
    }
    
    fprintf(fp_silu_coe, ",");
    freeTensorQ(lut_y);
#endif
}

int silu_gen_eq(const char *name)
{
#ifdef SILU_GEN
    struct Tensor *w;
    struct TensorQ *wq;
    struct Tensor *b;
    struct TensorQ *bq;
    struct Tensor *t;
    float *sx;
    float *sw;
    float *so;
    long long *zx;
    long long *zw;
    long long *zo;
    double factor;
    double factor2;
    char *tdataint8;
    int n;
    int m;
    static char buf[1000];
    sprintf(buf, "0x%x:%s\n", silu_addr, name);
    fwrite(buf, 1, strlen(buf), fp_silu_addr);
    silu_addr+=1;
    for (int i = 255; i >= 0; i--)
    {
        // if(i==253) printf("253(-3)->%d\n",(char)(lut_y->data[i]&0xff));
        fprintf(fp_silu_coe, "%02x", i & 0xff);
    }
    for (int i = 0; i <256 ; i++)
    {
        fwrite(&i,1,1,fp_silu_image);
    }
    
    fprintf(fp_silu_coe, ",");
#endif
}

//padded to 32,transposed
int input_gen(struct TensorQ *input)
{
#ifdef INPUT_GEN
    fp_input_image = fopen("input/input.image", "wb+");
    struct TensorQ *it = transposeQ(input, 0, 2, 3, 1);
    int n = it->lens[3];
    int n32 = ((n + 31) >> 5) << 5;
    int tlen = getlengthQ(it);
    int it_data_offset = 0;
    int rem = tlen;
    static char buf32[32];
    int sn = 0;
    while (rem)
    {
        int cplen = __min(32, n - sn);
        for (int k = 0; k < 32; k++)
        {
            buf32[k] = 0;
        }
        for (int k = 0; k < cplen; k++)
        {
            buf32[k] = it->data[it_data_offset++];
        }
        fwrite(buf32, 32, 1, fp_input_image);
        sn = sn + 32 >= n ? 0 : sn + 32;
        rem -= cplen;
    }
    freeTensorQ(it);
#endif
}

//output without padding to 32,but already transposed
// int output_gen_x(struct TensorQ *node_quantized, const char* node_name)
// {
//     FILE *fptest0=fopen(node_name, "wb+");
//     struct TensorQ *testTensorQ=transposeQ(node_quantized,0,2,3,1);
//     //printshapeQ(node_quantized);
//     int tlen=getlengthQ(testTensorQ);
//     for(int i=0;i<tlen;i++){
//         //fprintf(fptest0,"%x\n",testTensorQ->data[i]);
//         fwrite(testTensorQ->data+i,1,1,fptest0);
//     }
// }

int output_gen_x(struct TensorQ *node_quantized, const char *filename)
{
    struct TensorQ *it = transposeQ(node_quantized, 0, 2, 3, 1);
    fp_input_image = fopen(filename, "wb+");
    int n = it->lens[3];
    int n32 = ((n + 31) >> 5) << 5;
    int tlen = getlengthQ(it);
    int it_data_offset = 0;
    int rem = tlen;
    static char buf32[32];
    int sn = 0;
    while (rem)
    {
        int cplen = __min(32, n - sn);
        for (int k = 0; k < cplen; k++)
        {
            buf32[k] = it->data[it_data_offset++];
        }
        fwrite(buf32, 32, 1, fp_input_image);
        sn = sn + 32 >= n ? 0 : sn + 32;
        rem -= cplen;
    }
    freeTensorQ(it);
}

int output_gen_float(struct Tensor *node, const char* node_name){
    FILE *fp_float=fopen(node_name, "wb+");
    int tlen=getlength(node);
    for(int i=0;i<tlen;i++){
        fwrite(node->data+i,4,1,fp_float);
    }
}

int output_gen_x_no_transpose(struct TensorQ *node_quantized, const char *filename)
{
    FILE *fptest1=fopen(filename, "wb+");
    int tlen=getlengthQ(node_quantized);
    for(int i=0;i<tlen;i++){
        fwrite(node_quantized->data+i,1,1,fptest1);
    }
}

int output_TensorQ(struct TensorQ *tensor,const char *filename)
{
    FILE *fptest0=fopen(filename,"wb+");
    //printshapeQ(node_quantized);
    int tlen=getlengthQ(tensor);
    for(int i=0;i<tlen;i++){
        //fprintf(fptest0,"%x\n",testTensorQ->data[i]);
        fwrite(tensor->data+i,1,1,fptest0);
    }
}



int main()
{
#ifdef BIAS_GEN
    fp_bias_coe = fopen("bias/bias.coe", "wb+");
    fp_bias_addr = fopen("bias/bias_addr.txt", "wb+");
    fp_bias_image = fopen("bias/bias.image","wb+");
    bias_addr = 0;
    static char buf[1000] = "memory_initialization_radix = 16;\nmemory_initialization_vector=\n";
    fwrite(buf, strlen(buf), 1, fp_bias_coe);
#endif
#ifdef WEIGHT_GEN
    fp_weight_image = fopen("weight/weight.image", "wb+");
    fp_weight_addr = fopen("weight/weight_offset.txt", "wb+");
    weight_addr = 0;
#endif
#ifdef SILU_GEN
    fp_silu_addr = fopen("act/silu_addr.txt", "wb+");
    fp_silu_coe = fopen("act/silu.coe", "wb+");
    fp_silu_image = fopen("act/silu.image","wb+");
    static char bufs[1000] = "memory_initialization_radix = 16;\nmemory_initialization_vector=\n";
    fwrite(bufs, strlen(bufs), 1, fp_silu_coe);
    silu_addr = 0;
#endif


    FILE *instruction_out=fopen("instruction/instruction.image", "wb+");;
    FILE *instruction_foo=fopen("instruction/foo.image", "wb+");;
    FILE *instruction_hex=fopen("instruction/instruction_hex.txt", "wb+");;

    // 1.input an image
    // image.raw can be a pre-processed image using python or other tools
    // it has shape of 640x640x1,has been normalized(its value falls between 0-1)
    int width = 1024;
    int height = 1024;
    DTYPE *tdata = (DTYPE *)readfile("image.bin");


    struct TensorQ *nodeimages_quantized = getTensorQ(4);
    nodeimages_quantized->lens[0] = 1;
    nodeimages_quantized->lens[1] = 3;
    nodeimages_quantized->lens[2] = 224;
    nodeimages_quantized->lens[3] = 224;
    int tlen = getlengthQ(nodeimages_quantized);
    nodeimages_quantized->data = (int *)malloc(sizeof(int) * tlen);
    float *s0 = (float *)readfile("initializer/input_scale");
    int *z0 = (int *)readfile("initializer/input_zero_point");
    for(int i=0;i<tlen;i++)
    {
        nodeimages_quantized->data[i] = nearbyintf((double)tdata[i] / *s0) +*z0; 
    }
    cliptensorQ(nodeimages_quantized);

    input_gen(nodeimages_quantized);
    output_gen_x_no_transpose(nodeimages_quantized, "nodeimages_quantized");

    // 3.inference
    struct Tensor *w;
    struct TensorQ *wq;
    struct Tensor *b;
    struct TensorQ *bq;
    struct Tensor *t;
    struct TensorQ *input_sig;
    float *sx;
    float *sw;
    float *so;
    long long *zx;
    long long *zw;
    long long *zo;
    double factor;
    double factor2;
    char *tdataint8;
    int n;
    int m;
    int tX[3];

    // read cfg
    char param_buffer[256];


    FILE *name_cfg;
    char *name_params[268];
    int name_param_idx = 0;
    name_cfg = fopen("./parameters/names.txt", "r");
    if(name_cfg == NULL){
        return 1;
    }

    while(fgets(param_buffer, sizeof(param_buffer), name_cfg) != NULL && name_param_idx < 268){
        param_buffer[strcspn(param_buffer, "\n")] = 0;
        name_params[name_param_idx] = strdup(param_buffer);
        name_param_idx++;
    }
    name_param_idx=0;


    FILE *scale_cfg;
    char *scale_params[331];
    int scale_param_idx = 0;
    scale_cfg = fopen("./parameters/scales.txt", "r");
    if(scale_cfg == NULL){
        return 1;
    }

    while(fgets(param_buffer, sizeof(param_buffer), scale_cfg) != NULL && scale_param_idx < 331){
        param_buffer[strcspn(param_buffer, "\n")] = 0;
        scale_params[scale_param_idx] = strdup(param_buffer);
        scale_param_idx++;
    }
    scale_param_idx=0;

    FILE *zero_point_cfg;
    char *zero_point_params[331];
    int zero_point_param_idx = 0;
    zero_point_cfg = fopen("./parameters/zero_points.txt", "r");
    if(zero_point_cfg == NULL){
        return 1;
    }

    while(fgets(param_buffer, sizeof(param_buffer), zero_point_cfg) != NULL && zero_point_param_idx < 331){
        param_buffer[strcspn(param_buffer, "\n")] = 0;
        zero_point_params[zero_point_param_idx] = strdup(param_buffer);
        zero_point_param_idx++;
    }
    zero_point_param_idx=0;
    
    FILE *quantized_cfg;
    char *quantized_params[164];
    int quantized_param_idx = 0;
    quantized_cfg = fopen("./parameters/quantized.txt", "r");
    if(quantized_cfg == NULL){
        return 1;
    }

    while(fgets(param_buffer, sizeof(param_buffer), quantized_cfg) != NULL && quantized_param_idx < 164){
        param_buffer[strcspn(param_buffer, "\n")] = 0;
        quantized_params[quantized_param_idx] = strdup(param_buffer);
        quantized_param_idx++;
    }
    quantized_param_idx=0;

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x0, 0x10000000,
        "Conv_0_quant",nodeimages_quantized,
        "initializer/input_scale", "initializer/input_zero_point",
        "initializer/193_quantized", "initializer/193_scale",
        "initializer/193_zero_point",
        "initializer/192_scale", 
        "initializer/192_zero_point", 
        "initializer/194_quantized",
        node192_quantized, 1, 7, 3, 2
    )

    silu_gen_eq("silu");

    QMaxPool(
        instruction_out, instruction_hex, 64, 1, 112, 64, 0, 0x10000000, 0x11000000,
        "maxpool",
        node192_quantized,
        node126_quantized, 3, 1, 2
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_3_quant",node126_quantized,
        "initializer/192_scale", 
        "initializer/192_zero_point", 
        "initializer/196_quantized", 
        "initializer/196_scale",
        "initializer/196_zero_point",
        "initializer/195_scale", 
        "initializer/195_zero_point", 
        "initializer/197_quantized",
        node195_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x12000000, 0x13000000,
        "Conv_5_quant",node195_quantized,
        "initializer/195_scale", 
        "initializer/195_zero_point", 
        "initializer/199_quantized", 
        "initializer/199_scale",
        "initializer/199_zero_point",
        "initializer/198_scale", 
        "initializer/198_zero_point", 
        "initializer/200_quantized",
        node198_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearAdd(
        instruction_out, instruction_hex, 64, 1, 56, 64, 0,
        0x13000000, 0x11000000, 0x10000000,
        "QLinearAdd",
        node198_quantized,
        "initializer/198_scale", 
        "initializer/198_zero_point", 
        node126_quantized,
        "initializer/192_scale", 
        "initializer/192_zero_point", 
        "initializer/132_scale", 
        "initializer/132_zero_point",
        node132_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex,  0x10000000, 0x11000000,
        "Conv_8_quant",node132_quantized,
        "initializer/132_scale", 
        "initializer/132_zero_point", 
        "initializer/202_quantized", 
        "initializer/202_scale",
        "initializer/202_zero_point",
        "initializer/201_scale", 
        "initializer/201_zero_point", 
        "initializer/203_quantized",
        node201_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_10_quant",node201_quantized,
        "initializer/201_scale", 
        "initializer/201_zero_point", 
        "initializer/205_quantized", 
        "initializer/205_scale",
        "initializer/205_zero_point",
        "initializer/204_scale", 
        "initializer/204_zero_point", 
        "initializer/206_quantized",
        node204_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearAdd(
        instruction_out, instruction_hex, 64, 1, 56, 65, 0, 0x12000000, 0x10000000, 0x11000000,
        "Add_11_quant", node204_quantized,
        "initializer/204_scale", 
        "initializer/204_zero_point",
        node132_quantized,
        "initializer/132_scale", 
        "initializer/132_zero_point",
        "initializer/139_scale", 
        "initializer/139_zero_point",  
        node139_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_13_quant",node139_quantized,
        "initializer/139_scale", 
        "initializer/139_zero_point", 
        "initializer/208_quantized", 
        "initializer/208_scale",
        "initializer/208_zero_point",
        "initializer/207_scale", 
        "initializer/207_zero_point", 
        "initializer/209_quantized",
        node207_quantized, 1, 3, 1, 2
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x14000000,
        "Conv_16_quant",node139_quantized,
        "initializer/139_scale", 
        "initializer/139_zero_point", 
        "initializer/214_quantized", 
        "initializer/214_scale",
        "initializer/214_zero_point",
        "initializer/213_scale", 
        "initializer/213_zero_point", 
        "initializer/215_quantized",
        node213_quantized, 1, 1, 0, 2
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x12000000, 0x13000000,
        "Conv_15_quant",node207_quantized,
        "initializer/207_scale", 
        "initializer/207_zero_point", 
        "initializer/211_quantized", 
        "initializer/211_scale",
        "initializer/211_zero_point",
        "initializer/210_scale", 
        "initializer/210_zero_point", 
        "initializer/212_quantized",
        node210_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearAdd(
        instruction_out, instruction_hex, 128, 1, 28, 28, 0, 0x13000000, 0x14000000, 0x10000000,
        "Add_17",
        node210_quantized,
        "initializer/210_scale",
        "initializer/210_zero_point",
        node213_quantized,
        "initializer/213_scale",
        "initializer/213_zero_point",
        "initializer/148_scale",
        "initializer/148_zero_point",
        node148_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x11000000,
        "Conv_19_quant",node148_quantized,
        "initializer/148_scale", 
        "initializer/148_zero_point", 
        "initializer/217_quantized", 
        "initializer/217_scale",
        "initializer/217_zero_point",
        "initializer/216_scale", 
        "initializer/216_zero_point", 
        "initializer/218_quantized",
        node216_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_21_quant",node216_quantized,
        "initializer/216_scale", 
        "initializer/216_zero_point", 
        "initializer/220_quantized", 
        "initializer/220_scale",
        "initializer/220_zero_point",
        "initializer/219_scale", 
        "initializer/219_zero_point", 
        "initializer/221_quantized",
        node219_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    
    QLinearAdd(
        instruction_out, instruction_hex, 128, 1, 28, 128, 0, 0x12000000, 0x10000000, 0x11000000,
        "Add_22",
        node219_quantized,
        "initializer/219_scale", 
        "initializer/219_zero_point", 
        node148_quantized,
        "initializer/148_scale", 
        "initializer/148_zero_point", 
        "initializer/155_scale", 
        "initializer/155_zero_point", 
        node155_quantized
    )


    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x10000000,
        "Conv_24_quant",node155_quantized,
        "initializer/155_scale", 
        "initializer/155_zero_point", 
        "initializer/223_quantized", 
        "initializer/223_scale",
        "initializer/223_zero_point",
        "initializer/222_scale", 
        "initializer/222_zero_point", 
        "initializer/224_quantized",
        node222_quantized, 1, 3, 1, 2
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_27_quant",node155_quantized,
        "initializer/155_scale", 
        "initializer/155_zero_point", 
        "initializer/229_quantized", 
        "initializer/229_scale",
        "initializer/229_zero_point",
        "initializer/228_scale", 
        "initializer/228_zero_point", 
        "initializer/230_quantized",
        node228_quantized, 1, 1, 0, 2
    )

    silu_gen_eq("silu");


    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x11000000,
        "Conv_26_quant",node222_quantized,
        "initializer/222_scale", 
        "initializer/222_zero_point", 
        "initializer/226_quantized", 
        "initializer/226_scale",
        "initializer/226_zero_point",
        "initializer/225_scale", 
        "initializer/225_zero_point", 
        "initializer/227_quantized",
        node225_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearAdd(
        instruction_out, instruction_hex, 256, 1, 14, 256, 0, 0x11000000, 0x12000000, 0x10000000,
        "Add_28",
        node225_quantized,
        "initializer/225_scale", 
        "initializer/225_zero_point",
        node228_quantized,
        "initializer/228_scale", 
        "initializer/228_zero_point",
        "initializer/164_scale", 
        "initializer/164_zero_point",
        node164_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x11000000,
        "Conv_30_quant",node164_quantized,
        "initializer/164_scale", 
        "initializer/164_zero_point", 
        "initializer/232_quantized", 
        "initializer/232_scale",
        "initializer/232_zero_point",
        "initializer/231_scale", 
        "initializer/231_zero_point", 
        "initializer/233_quantized",
        node231_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_32_quant",node231_quantized,
        "initializer/231_scale", 
        "initializer/231_zero_point", 
        "initializer/235_quantized", 
        "initializer/235_scale",
        "initializer/235_zero_point",
        "initializer/234_scale", 
        "initializer/234_zero_point", 
        "initializer/236_quantized",
        node234_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");


    QLinearAdd(
        instruction_out, instruction_hex, 256, 1, 14, 256, 0, 0x12000000, 0x10000000, 0x11000000,
        "Add_33",
        node234_quantized,
        "initializer/234_scale", 
        "initializer/234_zero_point", 
        node164_quantized,
        "initializer/164_scale", 
        "initializer/164_zero_point", 
        "initializer/171_scale", 
        "initializer/171_zero_point", 
        node171_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x10000000,
        "Conv_35_quant",node171_quantized,
        "initializer/171_scale", 
        "initializer/171_zero_point", 
        "initializer/238_quantized", 
        "initializer/238_scale",
        "initializer/238_zero_point",
        "initializer/237_scale", 
        "initializer/237_zero_point", 
        "initializer/239_quantized",
        node237_quantized, 1, 3, 1, 2
    )

    silu_gen_eq("silu");

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x11000000,
        "Conv_38_quant",node237_quantized,
        "initializer/171_scale", 
        "initializer/171_zero_point", 
        "initializer/244_quantized", 
        "initializer/244_scale",
        "initializer/244_zero_point",
        "initializer/243_scale", 
        "initializer/243_zero_point", 
        "initializer/245_quantized",
        node243_quantized, 1, 1, 0, 2
    )

    silu_gen_eq("silu");


    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x12000000,
        "Conv_37_quant",node237_quantized,
        "initializer/237_scale", 
        "initializer/237_zero_point", 
        "initializer/241_quantized", 
        "initializer/241_scale",
        "initializer/241_zero_point",
        "initializer/240_scale", 
        "initializer/240_zero_point", 
        "initializer/242_quantized",
        node240_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");


    QLinearAdd(
        instruction_out, instruction_hex, 512, 1, 7, 512, 0, 0x12000000, 0x11000000, 0x10000000,
        "Add_39",
        node240_quantized,
        "initializer/240_scale", 
        "initializer/240_zero_point", 
        node243_quantized,
        "initializer/243_scale", 
        "initializer/243_zero_point", 
        "initializer/180_scale", 
        "initializer/180_zero_point", 
        node180_quantized
    )

    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x10000000, 0x11000000,
        "Conv_41_quant", node180_quantized,
        "initializer/180_scale", 
        "initializer/180_zero_point", 
        "initializer/247_quantized", 
        "initializer/247_scale",
        "initializer/247_zero_point",
        "initializer/246_scale", 
        "initializer/246_zero_point", 
        "initializer/248_quantized",
        node246_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");


    QLinearConv_AUTO(
        instruction_out, instruction_hex, 0x11000000, 0x12000000,
        "Conv_43_quant", node246_quantized,
        "initializer/246_scale", 
        "initializer/246_zero_point", 
        "initializer/250_quantized", 
        "initializer/250_scale",
        "initializer/250_zero_point",
        "initializer/249_scale", 
        "initializer/249_zero_point", 
        "initializer/251_quantized",
        node249_quantized, 1, 3, 1, 1
    )

    silu_gen_eq("silu");    

    QLinearAdd(
        instruction_out, instruction_hex, 512, 1, 7, 512, 0, 0x12000000, 0x10000000, 0x11000000,
        "Add_44",
        node249_quantized,
        "initializer/249_scale", 
        "initializer/249_zero_point", 
        node180_quantized,
        "initializer/180_scale", 
        "initializer/180_zero_point",
        "initializer/187_scale", 
        "initializer/187_zero_point", 
        node187_quantized
    )


    
    



}
