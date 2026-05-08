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
    nodeimages_quantized->lens[1] = 1;
    nodeimages_quantized->lens[2] = 1024;
    nodeimages_quantized->lens[3] = 1024;
    int tlen = getlengthQ(nodeimages_quantized);
    nodeimages_quantized->data = (int *)malloc(sizeof(int) * tlen);
    float *s0 = (float *)readfile("initializer/images_scale");
    int *z0 = (int *)readfile("initializer/images_zero_point");
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
        instruction_out, instruction_hex,0x0, 0x10000000,
        name_params[name_param_idx],
        nodeimages_quantized,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node1,1,3,1,2);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node2
    );


    QLinearConv_AUTO(
        instruction_out, instruction_hex,0x10000000, 0x11000000,
        name_params[name_param_idx],
        node2,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node3,1,3,1,2);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node3,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node4
    );

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x12000000,
        name_params[name_param_idx],
        node4,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node5_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node5_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node6_split0
    );

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x13000000,
        name_params[name_param_idx],
        node4,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node5_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node5_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node6_split1
    );

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        node6_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node7,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node7,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node8
    );

    QLinearConv_AUTO(
        instruction_out,0x14000000, 0x15000000,
        name_params[name_param_idx],
        node8,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node9,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node9,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node10
    );

    QLinearAdd(instruction_out, instruction_hex, 32, 4, 256, 32, 0, 0x13000000, 0x15000000, 0x16000000,
        "_model.2_m.0_Add_quant",
        node6_split1, "initializer/_model.2_cv1_act_Mul_output_0_scale", "initializer/_model.2_cv1_act_Mul_output_0_zero_point",
        node10, "initializer/_model.2_m.0_cv2_act_Mul_output_0_scale", "initializer/_model.2_m.0_cv2_act_Mul_output_0_zero_point",
        "initializer/_model.2_m.0_Add_output_0_scale", "initializer/_model.2_m.0_Add_output_0_zero_point",
        node11
    );

    QLinearConcat(instruction_out, instruction_hex,32,4,256,32,64,0x12000000,0x13000000,0x17000000,
        "_model.2_Concat_quant",
        "initializer/_model.2_Concat_output_0_scale","initializer/_model.2_Concat_output_0_zero_point",
        node6_split0,
        "initializer/_model.2_cv1_act_Mul_output_0_scale","initializer/_model.2_cv1_act_Mul_output_0_zero_point",
        node6_split1,
        "initializer/_model.2_cv1_act_Mul_output_0_scale","initializer/_model.2_cv1_act_Mul_output_0_zero_point",
        node6_split0_split1,
        1
    );

    QLinearConcat(instruction_out,32,4,256,64,96,0x17000000,0x16000000,0x10000000,
        "_model.2_Concat_quant_1",
        "initializer/_model.2_Concat_output_0_scale","initializer/_model.2_Concat_output_0_zero_point",
        node6_split0_split1,
        "initializer/_model.2_Concat_output_0_scale","initializer/_model.2_Concat_output_0_zero_point",
        node11,
        "initializer/_model.2_m.0_Add_output_0_scale", "initializer/_model.2_m.0_Add_output_0_zero_point",
        node12,
        1
    );
    scale_param_idx+=1;
    zero_point_param_idx+=1;
    

    QLinearConv_AUTO(
        instruction_out, instruction_hex,0x10000000, 0x11000000,
        name_params[name_param_idx],
        node12,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node13,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node13,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node14
    );

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x12000000,
        name_params[name_param_idx],
        node14,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node15,1,3,1,2);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node15,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node16
    );

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x13000000,
        name_params[name_param_idx],
        node16,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node17_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node17_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node18_split0
    );

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x14000000,
        name_params[name_param_idx],
        node16,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node17_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node17_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node18_split1
    );

    QLinearConv_AUTO(
        instruction_out,0x14000000, 0x15000000,
        name_params[name_param_idx],
        node18_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node19,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node19,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node20
    );

    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        node20,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node21,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node21,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node22
    );

    QLinearAdd(
        instruction_out, instruction_hex,64,4, 128, 64, 0, 0x14000000, 0x16000000,0x17000000,
        "_model.4_m.9_Add_quant",
        node18_split1,"initializer/_model.4_cv1_act_Mul_output_0_scale","initializer/_model.4_cv1_act_Mul_output_0_zero_point",
        node22,"initializer/_model.4_m.0_cv2_act_Mul_output_0_scale","initializer/_model.4_m.0_cv2_act_Mul_output_0_zero_point",
        "initializer/_model.4_m.0_Add_output_0_scale","initializer/_model.4_m.0_Add_output_0_zero_point",
        node23
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x17000000, 0x18000000,
        name_params[name_param_idx],
        node23,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node24,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node24,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node25
    );

    QLinearConv_AUTO(
        instruction_out,0x18000000, 0x19000000,
        name_params[name_param_idx],
        node25,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node26,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node26,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node27
    );

    QLinearAdd(
        instruction_out, instruction_hex,64,2,128,64,0,0x17000000,0x19000000,0x1a000000,
        "_model.4_m.1_Add_quant",
        node23,"initializer/_model.4_m.0_Add_output_0_scale","initializer/_model.4_m.0_Add_output_0_zero_point",
        node27,"initializer/_model.4_m.1_cv2_act_Mul_output_0_scale","initializer/_model.4_m.1_cv2_act_Mul_output_0_zero_point",
        "initializer/_model.4_m.1_Add_output_0_scale","initializer/_model.4_m.1_Add_output_0_zero_point",
        node28
    );

    QLinearConcat(
        instruction_out, instruction_hex,64,2,128,64,128,0x13000000,0x14000000,0x10000000,
        "_model.4_Concat_quant_0",
        "initializer/_model.4_Concat_output_0_scale","initializer/_model.4_Concat_output_0_zero_point",
        node18_split0,"initializer/_model.4_cv1_act_Mul_output_0_scale","initializer/_model.4_cv1_act_Mul_output_0_zero_point",
        node18_split1,"initializer/_model.4_cv1_act_Mul_output_0_scale","initializer/_model.4_cv1_act_Mul_output_0_zero_point",
        node18_split0_split1,
        1
    );

    QLinearConcat(
        instruction_out,64,2,128,128,192,0x10000000,0x17000000,0x11000000,
        "_model.4_Concat_quant_1",
        "initializer/_model.4_Concat_output_0_scale","initializer/_model.4_Concat_output_0_zero_point",
        node18_split0_split1,"initializer/_model.4_Concat_output_0_scale","initializer/_model.4_Concat_output_0_zero_point",
        node23,"initializer/_model.4_m.0_Add_output_0_scale","initializer/_model.4_m.0_Add_output_0_zero_point",
        node18_split0_split1_23,
        1
    );

    QLinearConcat(
        instruction_out,64,2,128,192,256,0x11000000,0x1a000000,0x12000000,
        "_model.4_Concat_quant_2",
        "initializer/_model.4_Concat_output_0_scale","initializer/_model.4_Concat_output_0_zero_point",
        node18_split0_split1_23,"initializer/_model.4_Concat_output_0_scale","initializer/_model.4_Concat_output_0_zero_point",
        node28,"initializer/_model.4_m.1_Add_output_0_scale","initializer/_model.4_m.1_Add_output_0_zero_point",
        node29,
        1
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x1e000000,
        name_params[name_param_idx],
        node29,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node30,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node30,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node31
    );

    QLinearConv_AUTO(
        instruction_out,0x1e000000, 0x14000000,
        name_params[name_param_idx],
        node31,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node32,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node32,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node33
    );

    QLinearDepthConv(instruction_out, instruction_hex,32,2,64,256,0,0x14000000,0x15000000,
    "_model.5_cv2_conv_Conv_quant",
    node33,"initializer/_model.5_cv1_act_Mul_output_0_scale","initializer/_model.5_cv1_act_Mul_output_0_zero_point",
    "initializer/model.5.cv2.conv.weight_quantized","initializer/model.5.cv2.conv.weight_scale","initializer/model.5.cv2.conv.weight_zero_point",
    "initializer/_model.5_cv2_conv_Conv_output_0_scale","initializer/_model.5_cv2_conv_Conv_output_0_zero_point",
    "initializer/model.5.cv2.conv.bias_quantized",
    node34,1,3,1,2);



    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        node34,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        quantized_params[quantized_param_idx+1],
        node35_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node35_split0,
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        scale_params[scale_param_idx+5],zero_point_params[zero_point_param_idx+5],
        node36_split0
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;


    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x17000000,
        name_params[name_param_idx],
        node34,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node35_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node35_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node36_split1
    );

    QLinearConv_AUTO(
        instruction_out,0x17000000, 0x18000000,
        name_params[name_param_idx],
        node36_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node37,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node37,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node38
    );

    QLinearConv_AUTO(
        instruction_out,0x18000000, 0x19000000,
        name_params[name_param_idx],
        node38,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node39,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node39,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node40
    );

    QLinearAdd(instruction_out,128,4,64,128,0,0x17000000,0x19000000,0x1a000000,
    "_model.6_m.0_Add_quant",
    node36_split1,"initializer/_model.6_cv1_act_Mul_output_0_scale","initializer/_model.6_cv1_act_Mul_output_0_zero_point",
    node40,"initializer/_model.6_m.0_cv2_act_Mul_output_0_scale","initializer/_model.6_m.0_cv2_act_Mul_output_0_zero_point",
    "initializer/_model.6_m.0_Add_output_0_scale","initializer/_model.6_m.0_Add_output_0_zero_point",
    node41
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x1a000000, 0x1b000000,
        name_params[name_param_idx],
        node41,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node42,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node42,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node43
    );

    QLinearConv_AUTO(
        instruction_out,0x1b000000, 0x1c000000,
        name_params[name_param_idx],
        node43,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node44,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node44,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node45
    );

    QLinearAdd(instruction_out,128,4,64,128,0,0x1a000000,0x1c000000,0x1d000000,
    "_model.6_m.1_Add_quant",
    node41,"initializer/_model.6_m.0_Add_output_0_scale","initializer/_model.6_m.0_Add_output_0_zero_point",
    node45,"initializer/_model.6_m.1_cv2_act_Mul_output_0_scale","initializer/_model.6_m.1_cv2_act_Mul_output_0_zero_point",
    "initializer/_model.6_m.1_Add_output_0_scale","initializer/_model.6_m.1_Add_output_0_zero_point",
    node46
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;
    
    QLinearConcat(
        instruction_out, instruction_hex,128,4,64,128,256,0x16000000,0x17000000,0x10000000,
        "_model.6_Concat_quant_0",
        "initializer/_model.6_Concat_output_0_scale","initializer/_model.6_Concat_output_0_zero_point",
        node36_split0,"initializer/_model.6_cv1_act_Mul_output_0_scale","initializer/_model.6_cv1_act_Mul_output_0_zero_point",
        node36_split1,"initializer/_model.6_cv1_act_Mul_output_0_scale","initializer/_model.6_cv1_act_Mul_output_0_zero_point",
        node47_split0_split1,1
    );

    QLinearConcat(
        instruction_out, instruction_hex,128,4,64,256,384,0x10000000,0x1a000000,0x11000000,
        "_model.6_Concat_quant_1",
        "initializer/_model.6_Concat_output_0_scale","initializer/_model.6_Concat_output_0_zero_point",
        node47_split0_split1,"initializer/_model.6_Concat_output_0_scale","initializer/_model.6_Concat_output_0_zero_point",
        node41,"initializer/_model.6_m.0_Add_output_0_scale","initializer/_model.6_m.0_Add_output_0_zero_point",
        node47_split0_split1_41,1
    );

    QLinearConcat(
        instruction_out, instruction_hex,128,4,64,384,512,0x11000000,0x1d000000,0x12000000,
        "_model.6_Concat_quant_2",
        "initializer/_model.6_Concat_output_0_scale","initializer/_model.6_Concat_output_0_zero_point",
        node47_split0_split1_41,"initializer/_model.6_Concat_output_0_scale","initializer/_model.6_Concat_output_0_zero_point",
        node46,"initializer/_model.6_m.1_Add_output_0_scale","initializer/_model.6_m.1_Add_output_0_zero_point",
        node48,1
    );



    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x1f000000,
        name_params[name_param_idx],
        node48,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node49,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node49,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node50
    );


    QLinearConv_AUTO(
        instruction_out,0x1f000000, 0x14000000,
        name_params[name_param_idx],
        node50,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node51,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node51,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node52
    );

    QLinearDepthConv(instruction_out, instruction_hex,32,1,32,512,0,0x14000000,0x15000000,
        "_model.7_cv2_conv_Conv_quant",
        node52,"initializer/_model.7_cv1_act_Mul_output_0_scale","initializer/_model.7_cv1_act_Mul_output_0_zero_point",
        "initializer/model.7.cv2.conv.weight_quantized",
        "initializer/model.7.cv2.conv.weight_scale","initializer/model.7.cv2.conv.weight_zero_point",
        "initializer/_model.7_cv2_conv_Conv_output_0_scale","initializer/_model.7_cv2_conv_Conv_output_0_zero_point",
        "initializer/model.7.cv2.conv.bias_quantized",
        node53,1,3,1,2
    );

    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        node53,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        quantized_params[quantized_param_idx+1],
        node54_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node54_split0,
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        scale_params[scale_param_idx+5],zero_point_params[zero_point_param_idx+5],
        node55_split0
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;


    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x17000000,
        name_params[name_param_idx],
        node53,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node54_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node54_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node55_split1
    );

    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,256,0,0x17000000,0x18000000,
        "_model.8_m.0_cv1_cv1.0_conv_Conv_quant",
        node55_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node56,1,3,1,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node56,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node57
    );
    
    QLinearConv_AUTO(
        instruction_out,0x18000000, 0x19000000,
        name_params[name_param_idx],
        node57,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node58,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node58,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node59
    );

    QLinearDepthConvWithSiLU(
        instruction_out,32,1,32,512,0,0x19000000,0x1a000000,
        "_model.8_m.0_cv1_cv1.2_conv_Conv_quant",
        node59,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node60,1,7,3,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node60,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node61
    );

    QLinearConv_AUTO(
        instruction_out,0x1a000000, 0x1b000000,
        name_params[name_param_idx],
        node61,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node62,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node62,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node63
    );

    QLinearDepthConvWithSiLU(
        instruction_out,32,1,32,256,0,0x1b000000,0x1c000000,
        "_model.8_m.0_cv1_cv1.4_conv_Conv_quant",
        node63,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node64,1,3,1,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node64,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node65
    );

    QLinearAdd(instruction_out,256,1,32,256,0,0x17000000,0x1c000000,0x1d000000,
    "_model.8_m.0_Add_quant",
    node55_split1,"initializer/_model.8_cv1_act_Mul_output_0_scale","initializer/_model.8_cv1_act_Mul_output_0_zero_point",
    node65,"initializer/_model.8_m.0_cv1_cv1.4_act_Mul_output_0_scale","initializer/_model.8_m.0_cv1_cv1.4_act_Mul_output_0_zero_point",
    "initializer/_model.8_m.0_Add_output_0_scale","initializer/_model.8_m.0_Add_output_0_zero_point",
    node66
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConcat(
        instruction_out,256,2,64,256,512,0x16000000,0x17000000,0x10000000,
        "_model.8_Concat_quant_0",
        "initializer/_model.8_Concat_output_0_scale","initializer/_model.8_Concat_output_0_zero_point",
        node55_split0,"initializer/_model.8_cv1_act_Mul_output_0_scale","initializer/_model.8_cv1_act_Mul_output_0_zero_point",
        node55_split1,"initializer/_model.8_cv1_act_Mul_output_0_scale","initializer/_model.8_cv1_act_Mul_output_0_zero_point",
        node55_split0_split1,1
    );

    QLinearConcat(
        instruction_out,256,2,64,512,768,0x10000000,0x1d000000,0x11000000,
        "_model.8_Concat_quant_1",
        "initializer/_model.8_Concat_output_0_scale","initializer/_model.8_Concat_output_0_zero_point",
        node55_split0_split1,"initializer/_model.8_Concat_output_0_scale","initializer/_model.8_Concat_output_0_zero_point",
        node66,"initializer/_model.8_m.0_Add_output_0_scale","initializer/_model.8_m.0_Add_output_0_zero_point",
        node67,1
    );


    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x12000000,
        name_params[name_param_idx],
        node67,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node68,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node68,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node69
    );

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x13000000,
        name_params[name_param_idx],
        node69,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node70,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node70,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node71
    );

    QMaxPool(
        instruction_out,256,1,16,256,0,
        0x13000000,0x14000000,
        "MaxPool_0",
        node71,node72,5,2,1
    );
    QMaxPool(
        instruction_out,256,1,16,256,0,
        0x14000000,0x15000000,
        "MaxPool_1",
        node72,node73,5,2,1
    );
    QMaxPool(
        instruction_out,256,1,16,256,0,
        0x15000000,0x16000000,
        "MaxPool_2",
        node73,node74,5,2,1
    );

    QLinearConcat(
        instruction_out,256,1,32,256,512,
        0x13000000,0x14000000,0x10000000,
        "_model.9_Concat_quant_0",
        "initializer/_model.9_Concat_output_0_scale","initializer/_model.9_Concat_output_0_zero_point",
        node71,"initializer/_model.9_cv1_act_Mul_output_0_scale","initializer/_model.9_cv1_act_Mul_output_0_zero_point",
        node72,"initializer/_model.9_cv1_act_Mul_output_0_scale","initializer/_model.9_cv1_act_Mul_output_0_zero_point",
        node7172,1
    );

    QLinearConcat(
        instruction_out,256,1,32,512,768,
        0x10000000,0x15000000,0x11000000,
        "_model.9_Concat_quant_1",
        "initializer/_model.9_Concat_output_0_scale","initializer/_model.9_Concat_output_0_zero_point",
        node7172,"initializer/_model.9_Concat_output_0_scale","initializer/_model.9_Concat_output_0_zero_point",
        node73,"initializer/_model.9_cv1_act_Mul_output_0_scale","initializer/_model.9_cv1_act_Mul_output_0_zero_point",
        node717273,1
    );

    QLinearConcat(
        instruction_out,256,1,32,768,1024,
        0x11000000,0x16000000,0x12000000,
        "_model.9_Concat_quant_2",
        "initializer/_model.9_Concat_output_0_scale","initializer/_model.9_Concat_output_0_zero_point",
        node717273,"initializer/_model.9_Concat_output_0_scale","initializer/_model.9_Concat_output_0_zero_point",
        node74,"initializer/_model.9_cv1_act_Mul_output_0_scale","initializer/_model.9_cv1_act_Mul_output_0_zero_point",
        node75,1
    );



    scale_param_idx+=1;
    zero_point_param_idx+=1;
    name_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x13000000,
        name_params[name_param_idx],
        node75,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node76,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node76,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node77
    );    

    
    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        node77,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node78_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node78_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node79_split0
    );

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x15000000,
        name_params[name_param_idx],
        node77,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node78_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node78_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node79_split1
    );

    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        node79_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node80,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node80,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node81
    );    

    QLinearConv(
        instruction_out, instruction_hex,256,1,32,256,0,
        0x16000000,0x17000000,
        "_model.10_ffn_ffn.1_conv_Conv_quant",
        node81,scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],node82,1,1,0,1
    );
    scale_param_idx+=4;
    zero_point_param_idx+=4;
    quantized_param_idx+=2;

    silu_gen_eq("_model.10_ffn_ffn.1_conv_Conv_quant_no_silu");

    QLinearAdd(
        instruction_out,256,2,32,256,0,0x15000000,0x17000000,0x18000000,
        "_model.10_Add_quant",
        node79_split1,"initializer/_model.10_cv1_act_Mul_output_0_scale","initializer/_model.10_cv1_act_Mul_output_0_zero_point",
        node82,"initializer/_model.10_ffn_ffn.1_conv_Conv_output_0_scale","initializer/_model.10_ffn_ffn.1_conv_Conv_output_0_zero_point",
        "initializer/_model.10_Add_output_0_scale","initializer/_model.10_Add_output_0_zero_point",
        node83
    );


    QLinearConcat(
        instruction_out,256,4,32,256,512,0x14000000,0x18000000,0x10000000,
        "_model.10_Concat_quant",
        "initializer/_model.10_Concat_output_0_scale","initializer/_model.10_Concat_output_0_zero_point",
        node79_split0,"initializer/_model.10_cv1_act_Mul_output_0_scale","initializer/_model.10_cv1_act_Mul_output_0_zero_point",
        node83,"initializer/_model.10_Add_output_0_scale","initializer/_model.10_Add_output_0_zero_point",
        node84,1
    );



    QLinearConv_AUTO(
        instruction_out,0x10000000, 0x20000000,
        name_params[name_param_idx],
        node84,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node85,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node85,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node86
    );



    QResize(
        instruction_out,256,4,8,512,0,0x20000000,0x12000000,
        "_model.11_Resize",
        node86,"initializer/86", "initializer/ortshared_1_1_4_0_token_8", node87
    );

    QLinearConcat(
        instruction_out,256,2,32,512,768,
        0x12000000,0x1f000000,0x13000000,
        "_model.12_Concat_quant",
        "initializer/_model.12_Concat_output_0_scale","initializer/_model.12_Concat_output_0_zero_point",
        node87,
        "initializer/_model.10_cv2_act_Mul_output_0_scale","initializer/_model.10_cv2_act_Mul_output_0_zero_point",
        node50,
        "initializer/_model.6_cv2_act_Mul_output_0_scale","initializer/_model.6_cv2_act_Mul_output_0_zero_point",
        node88,1);

    scale_param_idx+=1;
    zero_point_param_idx+=1;



    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        node88,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node89_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node89_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node90_split0
    );

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x15000000,
        name_params[name_param_idx],
        node88,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node89_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node89_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node90_split1
    );

    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        node90_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node91,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node91,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node92
    );    

    QLinearConv_AUTO(
        instruction_out,0x16000000, 0x17000000,
        name_params[name_param_idx],
        node92,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node93,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node93,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node94
    );    

    QLinearConcat(
        instruction_out,128,1,64,128,256,
        0x14000000,0x15000000,0x10000000,
        "_model.13_Concat_quant_0",
        "initializer/_model.13_Concat_output_0_scale","initializer/_model.13_Concat_output_0_zero_point",
        node90_split0,"initializer/_model.13_cv1_act_Mul_output_0_scale","initializer/_model.13_cv1_act_Mul_output_0_zero_point",
        node90_split1,"initializer/_model.13_cv1_act_Mul_output_0_scale","initializer/_model.13_cv1_act_Mul_output_0_zero_point",
        node95_split0_split1,1
    );

    QLinearConcat(
        instruction_out,128,1,64,256,384,
        0x10000000,0x17000000,0x11000000,
        "_model.13_Concat_quant_1",
        "initializer/_model.13_Concat_output_0_scale","initializer/_model.13_Concat_output_0_zero_point",
        node95_split0_split1,"initializer/_model.13_Concat_output_0_scale","initializer/_model.13_Concat_output_0_zero_point",
        node94,"initializer/_model.13_m.0_cv2_act_Mul_output_0_scale","initializer/_model.13_m.0_cv2_act_Mul_output_0_zero_point",
        node96,1
    );


    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x1f000000,
        name_params[name_param_idx],
        node96,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node97,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node97,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node98
    );  


    //1e 31
    QResize(
        instruction_out,256,1,64,256,0,0x1f000000,0x13000000,
        "_model.14_Resize",
        node98,"98","initializer/ortshared_1_1_4_0_token_8",
        node99
    );


    QLinearConcat(
        instruction_out,128,2,64,256,384,
        0x13000000,0x1e000000,0x10000000,
        "_model.15_Concat_quant","initializer/_model.15_Concat_output_0_scale","initializer/_model.15_Concat_output_0_zero_point",
        node99,"initializer/_model.13_cv2_act_Mul_output_0_scale","initializer/_model.13_cv2_act_Mul_output_0_zero_point",
        node31,"initializer/_model.4_cv2_act_Mul_output_0_scale","initializer/_model.4_cv2_act_Mul_output_0_zero_point",
        node100,1
    )

    scale_param_idx+=1;
    zero_point_param_idx+=1;


    QLinearConv_AUTO(
        instruction_out, instruction_hex,0x10000000, 0x11000000,
        name_params[name_param_idx],
        node100,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node101_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], node101_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node102_split0
    );  



    QLinearConv_AUTO(
        instruction_out,0x10000000, 0x12000000,
        name_params[name_param_idx],
        node100,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node101_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node101_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node102_split1
    ); 

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x13000000,
        name_params[name_param_idx],
        node102_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node103,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node103,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node104
    );      

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        node104,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node105,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node105,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node106
    );    

    QLinearConcat(
        instruction_out,64,2,128,64,128,
        0x11000000,0x12000000,0x10000000,
        "_model.16_Concat_quant_0",
        "initializer/_model.16_Concat_output_0_scale","initializer/_model.16_Concat_output_0_zero_point",
        node102_split0,"initializer/_model.16_cv1_act_Mul_output_0_scale","initializer/_model.16_cv1_act_Mul_output_0_zero_point",
        node102_split1,"initializer/_model.16_cv1_act_Mul_output_0_scale","initializer/_model.16_cv1_act_Mul_output_0_zero_point",
        node107_split0_split1,1
    );

    QLinearConcat(
        instruction_out,64,2,128,128,192,
        0x10000000,0x14000000,0x11000000,
        "_model.16_Concat_quant_1",
        "initializer/_model.16_Concat_output_0_scale","initializer/_model.16_Concat_output_0_zero_point",
        node107_split0_split1,"initializer/_model.16_Concat_output_0_scale","initializer/_model.16_Concat_output_0_zero_point",
        node106,"initializer/_model.16_m.0_cv2_act_Mul_output_0_scale","initializer/_model.16_m.0_cv2_act_Mul_output_0_zero_point",
        node108,1
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;



    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x12000000,
        name_params[name_param_idx],
        node108,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        node109,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], node109,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        node110
    );    

    //OUTPUT1
    QLinearDepthConvWithSiLU(
        instruction_out,32,2,128,128,0,
        0x12000000,0x13000000,
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_conv_Conv_quant",
        node110,"initializer/_model.16_cv2_act_Mul_output_0_scale","initializer/_model.16_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.0.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.0.0.0.conv.weight_scale","initializer/model.23.one2one_cv3.0.0.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.0.0.conv.bias_quantized",
        model23_one2one_cv3_0_conv_Conv_output0_quantized,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Mul_quant",
        model23_one2one_cv3_0_conv_Conv_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Mul_output_0_zero_point",
        model23_one2one_cv3_0_act_Mul_output0_quantized);


    QLinearConv(
        instruction_out, instruction_hex,128,2,64,128,0,
        0x13000000,0x14000000,
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_conv_Conv_quant",
        model23_one2one_cv3_0_act_Mul_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.0.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.0.0.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.0.0.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.0.1.conv.bias_quantized",
        model23_one2one_cv3_0_0_1_conv_Conv_output0_quantized,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Sigmoid_quant",
        "_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Mul_quant",
        model23_one2one_cv3_0_0_1_conv_Conv_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Mul_output_0_zero_point",
        model23_one2one_cv3_0_0_1_act_Mul_output0_quantized);


    QLinearDepthConvWithSiLU(
        instruction_out,32,2,64,128,0,
        0x14000000,0x15000000,
        "_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_conv_Conv_quant",
        model23_one2one_cv3_0_0_1_act_Mul_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.0_one2one_cv3.0.0.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.1.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.0.1.0.conv.weight_scale",
        "initializer/model.23.one2one_cv3.0.1.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.1.0.conv.bias_quantized",
        model23_one2one_cv3_0_1_0_conv_Conv_output0_quantized,1,3,1,1);
    
    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.0.1_one2one_cv3.0.0_one2one_cv3.0.1.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.0.1_one2one_cv3.0.0_one2one_cv3.0.1.0_act_Mul_quant",
        model23_one2one_cv3_0_1_0_conv_Conv_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Mul_output_0_zero_point",
        model23_one2one_cv3_0_1_0_act_Mul_output0_quantized);

    QLinearConv(
        instruction_out, instruction_hex,128,2,64,128,0,
        0x15000000,0x16000000,
        "_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_conv_Conv_quant",
        model23_one2one_cv3_0_1_0_act_Mul_output0_quantized,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.1.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.0.1.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.0.1.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.1.1.conv.bias_quantized",
        mode23_3,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Sigmoid_quant",
        "_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Mul_quant",
        mode23_3,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Mul_output_0_zero_point",
        node23_4);

    QLinearConv(
        instruction_out,32,2,64,32,0,
        0x16000000,0x22100000,
        "_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_quant",
        node23_4,
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.1_one2one_cv3.0.1.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.0.2.weight_quantized",
        "initializer/model.23.one2one_cv3.0.2.weight_scale",
        "initializer/model.23.one2one_cv3.0.2.weight_zero_point",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_output_0_zero_point",
        "initializer/ortshared_1_1_1_quantized",
        model_23_conv_output_0,1,1,0,1
    );

    silu_gen_eq("_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_quant");
    output_gen_x(model_23_conv_output_0, "_model.23_one2one_cv3.0_one2one_cv3.0.2_Conv_output_0");

    //OUTPUT0
    //0x12000000
    QLinearConv(
        instruction_out,64,1,64,64,0,
        0x12000000,0x13000000,
        "_model.23_one2one_cv2.0_one2one_cv2.0.0_conv_Conv_quant",
        node110,
        "initializer/_model.16_cv2_act_Mul_output_0_scale","initializer/_model.16_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.0.0.conv.weight_scale","initializer/model.23.one2one_cv2.0.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_conv_Conv_output_0_scale","initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.0.conv.bias_quantized",
        mode23_cv2_0,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Sigmoid_quant",
        "_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Mul_quant",
        mode23_cv2_0,
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Mul_output_0_zero_point",
        mode23_cv2_1);

    QLinearConv(
        instruction_out,64,2,64,64,0,
        0x13000000,0x14000000,
        "_model.23_one2one_cv2.0_one2one_cv2.0.1_conv_Conv_quant",
        mode23_cv2_1,
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.0.1.conv.weight_scale",
        "initializer/model.23.one2one_cv2.0.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.1.conv.bias_quantized",
        mode23_cv2_2,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Sigmoid_quant",
        "_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Mul_quant",
        mode23_cv2_2,
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Mul_output_0_zero_point",
        mode23_cv2_3);


    QLinearConv(
        instruction_out,64,2,64,64,0,
        0x14000000,0x22000000,
        "_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_quant",
        mode23_cv2_3,
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.2.weight_quantized",
        "initializer/model.23.one2one_cv2.0.2.weight_scale",
        "initializer/model.23.one2one_cv2.0.2.weight_zero_point",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.0.2.bias_quantized",
        model_23_conv_cv2_output_0,1,1,0,1
    );
    silu_gen_eq("_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_quant_SiLU");
    output_gen_x(model_23_conv_cv2_output_0, "_model.23_one2one_cv2.0_one2one_cv2.0.2_Conv_output_0");


    //OUTOUT2
    scale_param_idx=218;
    zero_point_param_idx=218;
    quantized_param_idx=98;
    name_param_idx=145;

    QLinearConv_AUTO(
        instruction_out,0x12000000,0x13000000,
        name_params[name_param_idx],
        node110,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model17_0,1,3,1,2);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model17_0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model17_1
    );    


    // //0x1f000000 98 128+256
    QLinearConcat(
        instruction_out,128,2,64,128,384,
        0x13000000,0x1f000000,0x10000000,
        "initializer/_model.18_Concat_quant",
        "initializer/_model.18_Concat_output_0_scale",
        "initializer/_model.18_Concat_output_0_zero_point",
        model17_1,
        "initializer/_model.17_act_Mul_output_0_scale",
        "initializer/_model.17_act_Mul_output_0_zero_point",
        node98,
        "initializer/_model.13_cv2_act_Mul_output_0_scale",
        "initializer/_model.13_cv2_act_Mul_output_0_zero_point",
        model18_0,1
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out, instruction_hex,0x10000000, 0x11000000,
        name_params[name_param_idx],
        model18_0,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model19_0_split0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], model19_0_split0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model19_1_split0
    );  

    QLinearConv_AUTO(
        instruction_out,0x10000000, 0x12000000,
        name_params[name_param_idx],
        model18_0,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model19_0_split1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model19_0_split1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model19_1_split1
    ); 

    QLinearConv_AUTO(
        instruction_out,0x12000000, 0x13000000,
        name_params[name_param_idx],
        model19_1_split1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model19_2,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model19_2,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model19_3
    ); 

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        model19_3,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model19_4,1,3,1,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model19_4,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model19_5
    ); 


//"/model.19/m.0/cv2/act/Mul_output_0_scale","/model.19/m.0/cv2/act/Mul_output_0_zero_point",

    QLinearConcat(
        instruction_out,128,2,64,128,256,
        0x11000000,0x12000000,0x10000000,
        "_model.19_Concat_quant",
        "initializer/_model.19_Concat_output_0_scale","initializer/_model.19_Concat_output_0_zero_point",
        model19_1_split0,
        "initializer/_model.19_cv1_act_Mul_output_0_scale","initializer/_model.19_cv1_act_Mul_output_0_zero_point",
        model19_1_split1,
        "initializer/_model.19_cv1_act_Mul_output_0_scale","initializer/_model.19_cv1_act_Mul_output_0_zero_point",
        model19_concat_0,1
    );

    QLinearConcat(
        instruction_out,128,2,64,256,384,
        0x10000000,0x14000000,0x11000000,
        "_model.19_Concat_quant_1",
        "initializer/_model.19_Concat_output_0_scale","initializer/_model.19_Concat_output_0_zero_point",
        model19_concat_0,
        "initializer/_model.19_Concat_output_0_scale","initializer/_model.19_Concat_output_0_zero_point",
        model19_5,
        "initializer/_model.19_m.0_cv2_act_Mul_output_0_scale","initializer/_model.19_m.0_cv2_act_Mul_output_0_zero_point",
        model19_6,1
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x21000000,
        name_params[name_param_idx],
        model19_6,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model19_7,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model19_7,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model19_8
    ); 
//TODO

//MODEL20
    QLinearConv_AUTO(
        instruction_out,0x21000000, 0x13000000,
        name_params[name_param_idx],
        model19_8,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model20_0,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model20_0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model20_1
    ); 

    QLinearDepthConv(
        instruction_out,32,2,64,256,0,
        0x13000000,0x14000000,
        "_model.20_cv2_conv_Conv_quant",
        model20_1,
        "initializer/_model.20_cv1_act_Mul_output_0_scale",
        "initializer/_model.20_cv1_act_Mul_output_0_zero_point",
        "initializer/model.20.cv2.conv.weight_quantized",
        "initializer/model.20.cv2.conv.weight_scale",
        "initializer/model.20.cv2.conv.weight_zero_point",
        "initializer/_model.20_cv2_conv_Conv_output_0_scale",
        "initializer/_model.20_cv2_conv_Conv_output_0_zero_point",
        "initializer/model.20.cv2.conv.bias_quantized",
        model20_2,1,3,1,2
    );

    //20 node86 256+512
    QLinearConcat(
        instruction_out,256,2,32,256,768,
        0x14000000,0x20000000,0x10000000,
        "_model.21_Concat_quant",
        "initializer/_model.21_Concat_output_0_scale",
        "initializer/_model.21_Concat_output_0_zero_point",
        model20_2,
        "initializer/_model.20_cv2_conv_Conv_output_0_scale",
        "initializer/_model.20_cv2_conv_Conv_output_0_zero_point",
        node86,
        "initializer/_model.10_cv2_act_Mul_output_0_scale",
        "initializer/_model.10_cv2_act_Mul_output_0_zero_point",
        model21_concat,1
    );


    scale_param_idx=252;
    zero_point_param_idx=252;
    quantized_param_idx=114;
    name_param_idx=167;

    QLinearConv_AUTO(
        instruction_out, instruction_hex,0x10000000, 0x11000000,
        name_params[name_param_idx],
        model21_concat,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model21_split_0,1,1,0,1);

    QLinearSiLU_AUTO_SPLIT(
        name_params[name_param_idx], name_params[name_param_idx], model21_split_0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model21_1_split_0
    );  

    QLinearConv_AUTO(
        instruction_out,0x10000000, 0x12000000,
        name_params[name_param_idx],
        model21_concat,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model21_split_1,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model21_split_1,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model21_1_split_1
    ); 


    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,256,0,0x12000000,0x13000000,
        "_model.22_m.0_cv1_cv1.0_conv_Conv_quant",
        model21_1_split_1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_0,1,3,1,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_1
    );

    QLinearConv_AUTO(
        instruction_out,0x13000000, 0x14000000,
        name_params[name_param_idx],
        model22_1,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_2,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_2,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_3
    ); 

    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,512,0,0x14000000,0x15000000,
        "_model.22_m.0_cv1_cv1.2_conv_Conv_quant",
        model22_3,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_4,1,7,3,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_4,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_5
    );

    QLinearConv_AUTO(
        instruction_out,0x15000000, 0x16000000,
        name_params[name_param_idx],
        model22_5,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_6,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_6,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_7
    ); 

    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,256,0,0x16000000,0x17000000,
        "_model.22_m.0_cv1_cv1.4_conv_Conv_quant",
        model22_7,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_8,1,3,1,1
    );
    name_param_idx+=1;
    quantized_param_idx+=2;
    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_8,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_9
    );

    QLinearAdd(
        instruction_out,256,1,32,256,0,0x12000000,0x17000000,0x18000000,
        "_model.22_m.0_Add_quant",
        model21_1_split_1,
        "initializer/_model.22_cv1_act_Mul_output_0_scale",
        "initializer/_model.22_cv1_act_Mul_output_0_zero_point",
        model22_9,
        "initializer/_model.22_m.0_cv1_cv1.4_act_Mul_output_0_scale",
        "initializer/_model.22_m.0_cv1_cv1.4_act_Mul_output_0_zero_point",
        "initializer/_model.22_m.0_Add_output_0_scale",
        "initializer/_model.22_m.0_Add_output_0_zero_point",
        model22_10
    );
    

    QLinearConcat(
        instruction_out,256,2,32,256,512,
        0x11000000,0x12000000,0x10000000,
        "_model.22_Concat_quant",
        "initializer/_model.22_Concat_output_0_scale","initializer/_model.22_Concat_output_0_zero_point",
        model21_1_split_0,
        "initializer/_model.22_cv1_act_Mul_output_0_scale","initializer/_model.22_cv1_act_Mul_output_0_zero_point",
        model21_1_split_1,
        "initializer/_model.22_cv1_act_Mul_output_0_scale","initializer/_model.22_cv1_act_Mul_output_0_zero_point",
        model22_concat_0,1
    );

    QLinearConcat(
        instruction_out,256,2,32,512,768,
        0x10000000,0x18000000,0x11000000,
        "_model.22_Concat_quant",
        "initializer/_model.22_Concat_output_0_scale","initializer/_model.22_Concat_output_0_zero_point",
        model22_concat_0,
        "initializer/_model.22_Concat_output_0_scale","initializer/_model.22_Concat_output_0_zero_point",
        model22_10,
        "initializer/_model.22_m.0_Add_output_0_scale","initializer/_model.22_m.0_Add_output_0_zero_point",
        model22_concat,1
    );

    scale_param_idx+=1;
    zero_point_param_idx+=1;
    scale_param_idx+=1;
    zero_point_param_idx+=1;

    QLinearConv_AUTO(
        instruction_out,0x11000000, 0x10000000,
        name_params[name_param_idx],
        model22_concat,
        scale_params[scale_param_idx],zero_point_params[zero_point_param_idx],
        quantized_params[quantized_param_idx],
        scale_params[scale_param_idx+2],zero_point_params[zero_point_param_idx+2],
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        quantized_params[quantized_param_idx+1],
        model22_cv2_0,1,1,0,1);

    QLinearSiLU_AUTO(
        name_params[name_param_idx], name_params[name_param_idx], model22_cv2_0,
        scale_params[scale_param_idx+1],zero_point_params[zero_point_param_idx+1],
        scale_params[scale_param_idx+3],zero_point_params[zero_point_param_idx+3],
        scale_params[scale_param_idx+4],zero_point_params[zero_point_param_idx+4],
        model22_cv2_1
    ); 


    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,512,0,
        0x10000000,0x11000000,
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_conv_Conv_quant",
        model22_cv2_1,
        "initializer/_model.22_cv2_act_Mul_output_0_scale",
        "initializer/_model.22_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.0.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.2.0.0.conv.weight_scale",
        "initializer/model.23.one2one_cv3.2.0.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.0.0.conv.bias_quantized",
        model23_cv3_2_0,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Mul_quant",
        model23_cv3_2_0,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Mul_output_0_zero_point",
        model23_cv3_2_1);


    QLinearConv(
        instruction_out,128,1,64,128,0,
        0x11000000,0x12000000,
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_conv_Conv_quant",
        model23_cv3_2_1,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.0.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.2.0.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.2.0.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.0.1.conv.bias_quantized",
        model23_cv3_2_2,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Sigmoid_quant",
        "_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Mul_quant",
        model23_cv3_2_2,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Mul_output_0_zero_point",
        model23_cv3_2_3);


    QLinearDepthConvWithSiLU(
        instruction_out,32,1,64,128,0,
        0x12000000,0x13000000,
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_conv_Conv_quant",
        model23_cv3_2_3,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.0_one2one_cv3.2.0.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.1.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.2.1.0.conv.weight_scale",
        "initializer/model.23.one2one_cv3.2.1.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.1.0.conv.bias_quantized",
        model23_cv3_2_4,1,3,1,1);
    
    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Mul_quant",
        model23_cv3_2_4,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Mul_output_0_zero_point",
        model23_cv3_2_5);

    QLinearConv(
        instruction_out,128,1,64,128,0,
        0x13000000,0x14000000,
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_conv_Conv_quant",
        model23_cv3_2_5,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.1.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.2.1.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.2.1.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.1.1.conv.bias_quantized",
        model23_cv3_2_6,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Sigmoid_quant",
        "_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Mul_quant",
        model23_cv3_2_6,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Mul_output_0_zero_point",
        model23_cv3_2_7);

    QLinearConv(
        instruction_out,32,1,64,32,0,
        0x14000000,0x221F0000,
        "_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_quant",
        model23_cv3_2_7,
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.1_one2one_cv3.2.1.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.2.2.weight_quantized",
        "initializer/model.23.one2one_cv3.2.2.weight_scale",
        "initializer/model.23.one2one_cv3.2.2.weight_zero_point",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_output_0_zero_point",
        "initializer/ortshared_0_1_1_quantized",
        model23_cv3_2_8,1,1,0,1
    );
    silu_gen_eq("_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_quant");    
    output_gen_x(model23_cv3_2_8, "_model.23_one2one_cv3.2_one2one_cv3.2.2_Conv_output_0");

    QLinearConv(
        instruction_out,16,1,16,64,0,
        0x10000000,0x11000000,
        "_model.23_one2one_cv2.2_one2one_cv2.2.0_conv_Conv_quant",
        model22_cv2_1,
        "initializer/_model.22_cv2_act_Mul_output_0_scale",
        "initializer/_model.22_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.2.0.conv.weight_scale",
        "initializer/model.23.one2one_cv2.2.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.0.conv.bias_quantized",
        model23_cv22_0,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Sigmoid_quant",
        "_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Mul_quant",
        model23_cv22_0,
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Mul_output_0_zero_point",
        model23_cv22_1);

    
    QLinearConv(
        instruction_out,64,1,64,64,0,
        0x11000000,0x12000000,
        "_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_quant",
        model23_cv22_1,
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.2.1.conv.weight_scale",
        "initializer/model.23.one2one_cv2.2.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.1.conv.bias_quantized",
        model23_cv22_2,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_output_0_scale",
        "_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Mul_quant",
        model23_cv22_2,
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Mul_output_0_zero_point",
        model23_cv22_3);

    QLinearConv(
        instruction_out,64,1,64,64,0,
        0x12000000,0x221E0000,
        "_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_quant",
        model23_cv22_3,
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.2.weight_quantized",
        "initializer/model.23.one2one_cv2.2.2.weight_scale",
        "initializer/model.23.one2one_cv2.2.2.weight_zero_point",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.2.2.bias_quantized",
        model23_cv22_4,1,1,0,1
    );
    silu_gen_eq("_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_quant_SiLU");
    output_gen_x(model23_cv22_4, "_model.23_one2one_cv2.2_one2one_cv2.2.2_Conv_output_0");



    QLinearDepthConvWithSiLU(
        instruction_out,32,16,32,256,0,
        0x21000000,0x11000000,
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_conv_Conv_quant",
        model19_8,
        "initializer/_model.19_cv2_act_Mul_output_0_scale",
        "initializer/_model.19_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.0.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.1.0.0.conv.weight_scale",
        "initializer/model.23.one2one_cv3.1.0.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.0.0.conv.bias_quantized",
        model23_cv31_0,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Mul_quant",
        model23_cv31_0,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Mul_output_0_zero_point",
        model23_cv31_1);


    QLinearConv(
        instruction_out,128,1,64,128,0,
        0x11000000,0x12000000,
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_conv_Conv_quant",
        model23_cv31_1,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.0.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.1.0.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.1.0.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.0.1.conv.bias_quantized",
        model23_cv31_2,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Sigmoid_quant",
        "_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Mul_quant",
        model23_cv31_2,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Mul_output_0_zero_point",
        model23_cv31_3);


    QLinearDepthConvWithSiLU(
        instruction_out,32,1,64,128,0,
        0x12000000,0x13000000,
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_conv_Conv_quant",
        model23_cv31_3,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.0_one2one_cv3.1.0.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.1.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.1.1.0.conv.weight_scale",
        "initializer/model.23.one2one_cv3.1.1.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.1.0.conv.bias_quantized",
        model23_cv31_4,1,3,1,1);
    
    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Sigmoid_quant",
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Mul_quant",
        model23_cv31_4,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Mul_output_0_zero_point",
        model23_cv31_5);


    QLinearConv(
        instruction_out,128,1,64,128,0,
        0x13000000,0x14000000,
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_quant",
        model23_cv31_5,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.1.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv3.1.1.1.conv.weight_scale",
        "initializer/model.23.one2one_cv3.1.1.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.1.1.conv.bias_quantized",
        model23_cv31_6,1,1,0,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_output_0_scale",
        "_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Mul_quant",
        model23_cv31_6,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Mul_output_0_zero_point",
        model23_cv31_7);

    QLinearConv(
        instruction_out,32,1,64,32,0,
        0x14000000,0x221C0000,
        "_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_quant",
        model23_cv31_7,
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.1_one2one_cv3.1.1.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv3.1.2.weight_quantized",
        "initializer/model.23.one2one_cv3.1.2.weight_scale",
        "initializer/model.23.one2one_cv3.1.2.weight_zero_point",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_output_0_zero_point",
        "initializer/ortshared_2_1_1_quantized",
        model23_cv31_8,1,1,0,1
    );
    silu_gen_eq("_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_quant");  
    output_gen_x(model23_cv31_8, "_model.23_one2one_cv3.1_one2one_cv3.1.2_Conv_output_0");


    QLinearConv(
        instruction_out,32,1,32,64,0,
        0x21000000,0x11000000,
        "_model.23_one2one_cv2.1_one2one_cv2.1.0_conv_Conv_quant",
        model19_8,
        "initializer/_model.19_cv2_act_Mul_output_0_scale",
        "initializer/_model.19_cv2_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.0.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.1.0.conv.weight_scale",
        "initializer/model.23.one2one_cv2.1.0.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.0.conv.bias_quantized",
        model23_cv21_0,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Sigmoid_quant",
        "_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Mul_quant",
        model23_cv21_0,
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Mul_output_0_zero_point",
        model23_cv21_1);

    QLinearConv(
        instruction_out,64,1,64,64,0,
        0x11000000,0x12000000,
        "_model.23_one2one_cv2.1_one2one_cv2.1.1_conv_Conv_quant",
        model23_cv21_1,
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.0_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.1.conv.weight_quantized",
        "initializer/model.23.one2one_cv2.1.1.conv.weight_scale",
        "initializer/model.23.one2one_cv2.1.1.conv.weight_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_conv_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.1.conv.bias_quantized",
        model23_cv21_2,1,3,1,1);

    QLinearSiLU_AUTO(
        "_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Sigmoid_quant",
        "_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Mul_quant",
        model23_cv21_2,
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_conv_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_conv_Conv_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Sigmoid_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Sigmoid_output_0_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Mul_output_0_zero_point",
        model23_cv21_3);


    QLinearConv(
        instruction_out,64,1,64,64,0,
        0x12000000,0x22180000,
        "_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_quant",
        model23_cv21_3,
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Mul_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.1_act_Mul_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.2.weight_quantized",
        "initializer/model.23.one2one_cv2.1.2.weight_scale",
        "initializer/model.23.one2one_cv2.1.2.weight_zero_point",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_output_0_scale",
        "initializer/_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_output_0_zero_point",
        "initializer/model.23.one2one_cv2.1.2.bias_quantized",
        model23_cv21_4,1,1,0,1
    );
    silu_gen_eq("_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_quant");
    output_gen_x(model23_cv21_4, "_model.23_one2one_cv2.1_one2one_cv2.1.2_Conv_output_0");


    //2064384
    test_ff(instruction_out, 0x22000000, 0x1F80000);
    printf("\ndone.\n");
}
