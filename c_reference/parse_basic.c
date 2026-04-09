#pragma once
#include <stdio.h>
#define get_field(x) (x >> 3)
#define get_type(x) (x & 7)

void print_field_type(unsigned char x)
{
    printf("field=%d type=%d\n", get_field(x), get_type(x));
}

long long get_int64(FILE *fp)
{
    // we just assume there is no negetive value
    unsigned char t;
    long long ret = 0;
    int i = 0;
    do
    {
        fread(&t, 1, 1, fp);
        ret = ret | ((t & 0x7f) << (7 * i));
        i++;
    } while (t & 0x80);

    return ret;
}

char *get_string(FILE *fp)
{
    int len = get_int64(fp);
    char *str = (char *)malloc(len + 1);
    fread(str, 1, len, fp);
    str[len] = 0;
    return str;
}

char *get_bytes(FILE *fp,int len){
    char *str = (char *)malloc(len);
    fread(str, 1, len, fp);
    return str;
}