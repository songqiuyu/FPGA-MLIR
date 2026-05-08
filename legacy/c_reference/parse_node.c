#include<stdio.h>

void parse_node_attribute(FILE *fp,int len,char *name){
    int start=ftell(fp);
    int end=start+len;
    while(ftell(fp)<end){
        unsigned char t;
        fread(&t, 1, 1, fp);
        //printf("field type=%d\n",get_field(t));
        if(get_field(t)==1){
            char *attrname = get_string(fp);
            //printf("attrname=%s\n", attrname);
            free(attrname);
        }
        else if(get_field(t)==2){
            float fval;
            fread(&fval,1,4,fp);
            //printf("fval=%f\n",fval);
        }
        else if(get_field(t)==3){
            long long lval=get_int64(fp);
            //printf("lval=%lld\n",lval);
        }
        else if(get_field(t)==4){
            char *str = get_string(fp);
            //printf("bytes=%s\n", str);
            free(str);
        }
        else if(get_field(t)==5){ //tensor proto //this will output the initilizers either
            int len1=get_int64(fp);
            //printf("here we are name=%s\n",name);
            parse_tensor(fp,len1,name);
        }
        else if(get_field(t)==6){ //graph proto
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==7){
            float fval;
            fread(&fval,1,4,fp);
            //printf("fval7=%f\n",fval);
        }
        else if(get_field(t)==8){
            long long lval=get_int64(fp);
            //printf("lval8=%lld\n",lval);
        }
        else if(get_field(t)==9){
            char *str = get_string(fp);
            //printf("bytes=%s\n", str);
            free(str);
        }
        else if(get_field(t)==10){ //tensor proto
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==11){ //graph proto
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==13){ //graph proto
            char *doc = get_string(fp);
            free(doc);
        }
        else if(get_field(t)==14){ //type proto
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==15){ //type protos
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==20){ //attr type,regarded as int64
            int type=get_int64(fp);
            //printf("attr type=%d\n",type);
        }
        else if(get_field(t)==21){ //ref attr name
            char *str = get_string(fp);
            free(str);
        }
        else if(get_field(t)==22){ //sparse tensor
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else if(get_field(t)==23){ //sparse tensors
            int len1=get_int64(fp);
            fseek(fp,len1,SEEK_CUR);
        }
        else{
            //printf("attr %d unsupported\n",get_field(t));
            //exit(0);
        }
    }
}