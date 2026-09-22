#include "memvanta/llama_model.hpp"
#include <iostream>
#include <string>
#include <thread>

static void usage(){std::cout<<"memvanta_real --model model.gguf --prompt \"Hello\" [--n 64] [--threads N] [--ctx 2048] [--temperature 0]\n";}
int main(int argc,char**argv){try{std::string model,prompt="Hello";std::size_t n=64,ctx=0;unsigned threads=std::max(1u,std::thread::hardware_concurrency());float temp=0;for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(i+1>=argc)throw std::runtime_error("missing value for "+a);return std::string(argv[++i]);};if(a=="--model")model=val();else if(a=="--prompt")prompt=val();else if(a=="--n")n=std::stoull(val());else if(a=="--threads")threads=std::stoul(val());else if(a=="--ctx")ctx=std::stoull(val());else if(a=="--temperature")temp=std::stof(val());else if(a=="--help"){usage();return 0;}else throw std::runtime_error("unknown arg: "+a);}if(model.empty()){usage();return 1;}memvanta::LlamaModel m(model,threads,ctx);auto ids=m.tokenizer().encode(prompt,true);std::cout<<"[MemVanta real GGUF] prompt tokens="<<ids.size()<<" model="<<m.gguf().get_string("general.name").value_or("unknown")<<"\n";memvanta::Sampler s({temp,40,1234});auto out=m.generate(ids,n,s);for(int t:out)std::cout<<m.tokenizer().decode_piece(t)<<std::flush;std::cout<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}}
