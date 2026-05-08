#include <stdlib.h>
#include <stdio.h>
#include "parse_basic.c"
#include "parse_tensor.c"
#include "parse_node.c"


void parse_Node(FILE *fp, int len)
{
    int start = ftell(fp);
    int end = start + len;
    int isconstant=0;
    char *output=0;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            char *input = get_string(fp);
            //printf("input=%s\n", input);
            free(input);
        }
        else if (get_field(t) == 2)
        {
            output = get_string(fp);
        }
        else if (get_field(t) == 3)
        {
            char *name = get_string(fp);
            //printf("node name=%s\n", name);
            free(name);
        }
        else if (get_field(t) == 4)
        {
            char *optype = get_string(fp);
            //printf("optype=%s\n", optype);
            free(optype);
        }
        else if (get_field(t) == 5)
        {
            int len1 = get_int64(fp);
            parse_node_attribute(fp, len1,output);
        }
        else if (get_field(t) == 6)
        {
            char *doc = get_string(fp);
            // printf("doc=%s\n", doc);
            free(doc);
        }
        else if (get_field(t) == 7)
        {
            char *domain = get_string(fp);
            // printf("domain=%s\n", domain);
            free(domain);
        }
    }
    //printf("\n");

    if(output) free(output);
}



void parse_graph(FILE *fp, int len)
{
    int start = ftell(fp);
    int end = start + len;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            int len1 = get_int64(fp);
            parse_Node(fp, len1);//many many nodes
        }
        else if (get_field(t) == 2)
        {
            char *name = get_string(fp);
            // printf("graph name=%s\n", name);
            free(name);
        }
        else if (get_field(t) == 5)
        { // initializer or constant inputs,TensorProto
            int len1 = get_int64(fp);
            parse_tensor(fp, len1,0);
        }
        else if (get_field(t) == 10)
        {
            char *doc = get_string(fp);
            // printf("doc=%s\n", doc);
            free(doc);
        }
        else if (get_field(t) == 11)
        { // inputs of graph,ValueInfoProto
            int len1 = get_int64(fp);
            parse_valueinfo(fp, len1);
        }
        else if (get_field(t) == 12)
        { // outputs of graph
            int len1 = get_int64(fp);
            parse_valueinfo(fp, len1);
        }
        else if (get_field(t) == 13)
        { // value information
            int len1 = get_int64(fp);
            fseek(fp, len1, SEEK_CUR);
        }
        else if (get_field(t) == 14)
        { // quantization_annotation
            int len1 = get_int64(fp);
            fseek(fp, len1, SEEK_CUR);
        }
        else{
            printf("parse_graph unsupported\n");
            exit(0);
        }
    }
    printf("ftell=%d and end=%d\n", ftell(fp), end);
}

int main()
{
    FILE *fp = fopen("yolov5_gray_640.onnx", "rb+");
    unsigned char t;

    fread(&t, 1, 1, fp);
    print_field_type(t);
    long long ir_version = get_int64(fp);
    printf("ir version=%lld\n", ir_version);

    fread(&t, 1, 1, fp);
    print_field_type(t);
    char *producer = get_string(fp);
    printf("producer=%s\n", producer);
    free(producer);

    fread(&t, 1, 1, fp);
    print_field_type(t);
    char *prodcer_v = get_string(fp);
    printf("producer_v=%s\n", prodcer_v);
    free(prodcer_v);

    fread(&t, 1, 1, fp);
    print_field_type(t);
    int len = get_int64(fp);
    printf("graph length=%d\n", len);
    // goto graph layer
    parse_graph(fp, len);
}