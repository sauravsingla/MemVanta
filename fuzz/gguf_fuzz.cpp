#include "memvanta/gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
class TempFile {
public:
    TempFile(){
        char pattern[]="/tmp/memvanta-gguf-fuzz-XXXXXX";
        fd_=mkstemp(pattern);
        if(fd_<0) throw std::runtime_error("mkstemp failed");
        path_=pattern;
    }
    ~TempFile(){if(fd_>=0)::close(fd_);if(!path_.empty())std::remove(path_.c_str());}
    const std::string& path() const{return path_;}
    void write(const std::uint8_t* data,std::size_t size){
        if(ftruncate(fd_,static_cast<off_t>(size))!=0)throw std::runtime_error("ftruncate failed");
        std::size_t off=0;
        while(off<size){const auto n=pwrite(fd_,data+off,size-off,static_cast<off_t>(off));if(n<=0)throw std::runtime_error("pwrite failed");off+=static_cast<std::size_t>(n);}
    }
private:
    int fd_{-1};
    std::string path_;
};
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,std::size_t size){
    static TempFile file;
    try{
        file.write(data,size);
        memvanta::GgufFile parsed(file.path());
        for(const auto& tensor:parsed.tensors()){
            (void)tensor.elements();
            (void)parsed.tensor_data(tensor);
        }
    }catch(const std::exception&){
        // Invalid GGUF inputs are expected; crashes, sanitizer findings and hangs are not.
    }
    return 0;
}
