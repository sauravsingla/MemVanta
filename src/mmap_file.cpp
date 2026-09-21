#include "memvanta/mmap_file.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace memvanta {
MMapFile::MMapFile(const std::string& path){ open(path); }
MMapFile::~MMapFile(){ close(); }
MMapFile::MMapFile(MMapFile&& o) noexcept : fd_(o.fd_), data_(o.data_), size_(o.size_), advice_attempts_(o.advice_attempts_.load()), advice_failures_(o.advice_failures_.load()) { o.fd_=-1; o.data_=nullptr; o.size_=0; o.advice_attempts_.store(0); o.advice_failures_.store(0); }
MMapFile& MMapFile::operator=(MMapFile&& o) noexcept { if(this!=&o){ close(); fd_=o.fd_; data_=o.data_; size_=o.size_; advice_attempts_.store(o.advice_attempts_.load()); advice_failures_.store(o.advice_failures_.load()); o.fd_=-1;o.data_=nullptr;o.size_=0;o.advice_attempts_.store(0);o.advice_failures_.store(0);} return *this; }
void MMapFile::open(const std::string& path){
  close();advice_attempts_.store(0);advice_failures_.store(0);
  fd_=::open(path.c_str(),O_RDONLY);if(fd_<0)throw std::runtime_error("open failed: "+path);
  struct stat st{};if(fstat(fd_,&st)!=0){close();throw std::runtime_error("fstat failed");}
  if(st.st_size<=0){close();throw std::runtime_error("empty file");}
  if(static_cast<std::uintmax_t>(st.st_size)>std::numeric_limits<std::size_t>::max()){close();throw std::runtime_error("file too large for address space");}
  size_=static_cast<std::uint64_t>(st.st_size);
  void* p=mmap(nullptr,static_cast<std::size_t>(size_),PROT_READ,MAP_PRIVATE,fd_,0);if(p==MAP_FAILED){close();throw std::runtime_error("mmap failed");}
  data_=static_cast<std::byte*>(p);advise_range(0,size_,MADV_RANDOM);
}
void MMapFile::close(){if(data_){munmap(data_,static_cast<std::size_t>(size_));data_=nullptr;}if(fd_>=0){::close(fd_);fd_=-1;}size_=0;}
void MMapFile::advise_range(std::uint64_t off,std::uint64_t len,int advice) const {
  if(!data_||!len||off>=size_)return;
  std::lock_guard<std::mutex> advice_lock(advice_mu_);
  const long raw_page=sysconf(_SC_PAGESIZE);if(raw_page<=0){advice_attempts_.fetch_add(1);advice_failures_.fetch_add(1);return;}
  const auto page=static_cast<std::uint64_t>(raw_page);
  const auto end=off+std::min<std::uint64_t>(len,size_-off);
  const auto start_aligned=(off/page)*page;
  std::uint64_t end_aligned=end;
  const auto rem=end_aligned%page;if(rem&&end_aligned<=std::numeric_limits<std::uint64_t>::max()-(page-rem))end_aligned+=page-rem;
  end_aligned=std::min(end_aligned,size_);
  if(end_aligned<=start_aligned)return;
  advice_attempts_.fetch_add(1);
  if(madvise(data_+start_aligned,static_cast<std::size_t>(end_aligned-start_aligned),advice)!=0)advice_failures_.fetch_add(1);
}
void MMapFile::advise_sequential(std::uint64_t off,std::uint64_t len) const {advise_range(off,len,MADV_SEQUENTIAL);}
void MMapFile::advise_willneed(std::uint64_t off,std::uint64_t len) const {advise_range(off,len,MADV_WILLNEED);}
void MMapFile::advise_dontneed(std::uint64_t off,std::uint64_t len) const {advise_range(off,len,MADV_DONTNEED);}
}
