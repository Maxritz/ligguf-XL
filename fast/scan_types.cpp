#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include <map>
#include <direct.h>

static long long rd64(const uint8_t*& p){ long long v; memcpy(&v,p,8); p+=8; return v; }
static unsigned int rd32(const uint8_t*& p){ unsigned int v; memcpy(&v,p,4); p+=4; return v; }
static std::string rdstr(const uint8_t*& p){ long long n=rd64(p); std::string s((const char*)p,(size_t)n); p+=n; return s; }

static const char* tn(int t){
  switch(t){
  case 0:return "F32"; case 1:return "F16"; case 2:return "Q4_0"; case 3:return "Q4_1";
  case 6:return "Q5_0"; case 7:return "Q5_1"; case 8:return "Q8_0"; case 9:return "Q8_1";
  case 10:return "Q2_K"; case 11:return "Q3_K"; case 12:return "Q4_K"; case 13:return "Q5_K";
  case 14:return "Q6_K"; case 15:return "Q8_K"; case 16:return "IQ2_XXS"; case 17:return "IQ2_XS";
  case 18:return "IQ3_XXS"; case 19:return "IQ1_S"; case 20:return "IQ4_NL"; case 21:return "IQ3_S";
  case 22:return "IQ2_S"; case 23:return "IQ4_XS"; case 24:return "I8"; case 25:return "I16";
  case 26:return "I32"; case 27:return "I64"; case 28:return "F64"; case 29:return "IQ1_M";
  case 30:return "BF16"; case 34:return "TQ1_0"; case 35:return "TQ2_0"; case 39:return "MXFP4";
  case 40:return "NVFP4"; case 41:return "Q1_0"; case 42:return "Q2_0"; case 142:return "PQ2_0";
  default:return "??";
  }
}

int main(int argc,char** argv){
  for(int a=1;a<argc;a++){
    FILE* f=fopen(argv[a],"rb"); if(!f){printf("skip %s\n",argv[a]);continue;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint8_t> b(sz); fread(b.data(),1,sz,f); fclose(f);
    const uint8_t* p=b.data();
    if(memcmp(p,"GGUF",4)){printf("%s: not gguf\n",argv[a]);continue;}
    p+=4; unsigned ver=rd32(p); long long nten=rd64(p), nkv=rd64(p);
    std::string arch;
    std::map<int,int> types;
    long long total=0;
    for(long long i=0;i<nkv;i++){
      std::string k=rdstr(p); unsigned t=rd32(p);
      if(k=="general.architecture") arch=rdstr(p);
      else {
        // skip value
        const uint8_t* q=p;
        switch(t){
          case 0:case 1:case 2:case 3:case 4:case 5:case 6:case 7:case 10:case 11:case 12:case 13: q+=4; break;
          case 8:case 9: q+=8; break;
          case 14:{ long long n=rd64(q); for(long long j=0;j<n;j++){ unsigned et=rd32(q); q+=8; } break; }
          default: q+=8;
        }
        p=q;
      }
    }
    for(long long i=0;i<nten;i++){
      std::string k=rdstr(p); unsigned nd=rd32(p);
      std::vector<long long> d(nd); for(unsigned j=0;j<nd;j++) d[j]=rd64(p);
      unsigned t=rd32(p); long long off=rd64(p);
      types[t]++;
      long long els=1; for(unsigned j=0;j<nd;j++) els*=d[j];
      total+=els;
    }
    printf("%s\n  ver=%u arch=%s tensors=%I64d kv=%I64d elems=%I64d\n  types:",argv[a],ver,arch.c_str(),nten,nkv,total);
    for(std::map<int,int>::iterator it=types.begin();it!=types.end();++it)
      printf(" %s(%d)=%d",tn(it->first),it->first,it->second);
    printf("\n");
  }
  return 0;
}
