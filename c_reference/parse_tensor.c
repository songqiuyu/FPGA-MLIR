#include "parse_basic.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parse_tensor(FILE *fp, int len,char *name)
{ // FOR INITIALIZER
    int start = ftell(fp);
    int end = start + len;
    int nullname=name==0;
    char *data = 0;
    int len1;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            int dimi = get_int64(fp);
            // printf("dimi=%d\n",dimi);
        }
        else if (get_field(t) == 2)
        {
            int type = get_int64(fp);
            // printf("datatype=%d\n",type);
        }
        else if (get_field(t) == 3)
        {
            int len1 = get_int64(fp);
            fseek(fp, len1, SEEK_CUR);
        }
        else if (get_field(t) == 8)
        {
            if(!name)
            name = get_string(fp);
            //printf("tensor name=%s\n", name);
        }
        else if (get_field(t) == 9)
        {
            len1 = get_int64(fp);
            // printf("rawbytes len=%d\n", len1);
            data = get_bytes(fp, len1);
        }
        else
        {
            printf("t=%d\n", get_field(t));
            exit(0);
        }
    }
    // output to /initilizer/name
    if (name && data)
    {
        static char buf[100];
        sprintf(buf, "initializer/%s", name);
        FILE *fp1 = fopen(buf, "wb+");
        fwrite(data,1,len1,fp1);
        fclose(fp1);
    }
    if(name && nullname) free(name);
    if(data) free(data);

    
}

void parse_dimension(FILE *fp, int len)
{
    int start = ftell(fp);
    int end = start + len;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            int dim = get_int64(fp);
            //printf("dim=%d\n", dim);
        }
        else if (get_field(t) == 2)
        {
            char *str = get_string(fp);
            free(str);
        }
        else if (get_field(t) == 3)
        {
            char *str = get_string(fp);
            free(str);
        }
        else
        {
            printf("unsupported\n");
            exit(0);
        }
    }
}

void parse_tensorshape(FILE *fp, int len)
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
            // into the dimension layer
            parse_dimension(fp, len1);
        }
        else
        {
            printf("unsupported\n");
            exit(0);
        }
    }
}

void parse_tensor0(FILE *fp, int len)
{
    int start = ftell(fp);
    int end = start + len;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            int type = get_int64(fp);
            //printf("datatype=%d\n", type);
        }
        else if (get_field(t) == 2)
        {
            int len1 = get_int64(fp);
            parse_tensorshape(fp, len1);
        }
        else
        {
            printf("unsupported\n");
            exit(0);
        }
    }
}

void parse_type(FILE *fp, int len)
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
            parse_tensor0(fp, len1);
        }
        else
        {
            printf("unsupported\n");
            exit(0);
        }
    }
}

void parse_valueinfo(FILE *fp, int len)
{ // for inputs and outputs(but non initializer)
    int start = ftell(fp);
    int end = start + len;
    while (ftell(fp) < end)
    {
        unsigned char t;
        fread(&t, 1, 1, fp);
        if (get_field(t) == 1)
        {
            char *name = get_string(fp);
            //printf("valueinfo name=%s\n", name);
            free(name);
        }
        else if (get_field(t) == 2)
        {

            int len1 = get_int64(fp);
            parse_type(fp, len1);
        }
        else
        {
            int len1 = get_int64(fp);
            fseek(fp, len1, SEEK_CUR);
        }
    }
    //printf("\n");
}