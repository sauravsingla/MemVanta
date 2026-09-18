#include "memvanta/lru_cache.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
namespace memvanta {
TensorCache::TensorCache(std::uint64_t cap):capacity_(cap){}
void TensorCache::evict_for(std::uint64_t incoming){
  while(incoming>capacity_-used_ && !lru_.empty()){
    auto id=lru_.back(); auto it=map_.find(id);
    used_-=it->second.data->size(); lru_.pop_back(); map_.erase(it); stats_.evictions++;
  }
}
std::shared_ptr<const std::vector<std::byte>> TensorCache::get_or_load(std::uint32_t id,const std::byte* src,std::uint64_t bytes){
  std::lock_guard lk(mu_);
  auto it=map_.find(id);
  if(it!=map_.end()){
    stats_.hits++; lru_.erase(it->second.lru_it); lru_.push_front(id); it->second.lru_it=lru_.begin(); return it->second.data;
  }
  stats_.misses++;
  if(bytes>capacity_) throw std::runtime_error("tensor larger than cache capacity");
  if(bytes>std::numeric_limits<std::size_t>::max()) throw std::runtime_error("tensor too large for address space");
  if(bytes&&!src) throw std::runtime_error("null tensor source");
  evict_for(bytes);
  auto v=std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(bytes)); std::memcpy(v->data(),src,static_cast<std::size_t>(bytes)); stats_.bytes_copied+=bytes;
  lru_.push_front(id); map_.emplace(id,Entry{v,lru_.begin()}); used_+=bytes; return v;
}
bool TensorCache::contains(std::uint32_t id) const { std::lock_guard lk(mu_); return map_.contains(id); }
void TensorCache::insert_prefetched(std::uint32_t id,const std::byte* src,std::uint64_t bytes){
  std::lock_guard lk(mu_); if(map_.contains(id) || bytes>capacity_ || bytes>std::numeric_limits<std::size_t>::max()) return; if(bytes&&!src)throw std::runtime_error("null tensor source"); evict_for(bytes);
  auto v=std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(bytes)); std::memcpy(v->data(),src,static_cast<std::size_t>(bytes)); stats_.bytes_copied+=bytes;
  lru_.push_front(id); map_.emplace(id,Entry{v,lru_.begin()}); used_+=bytes;
}
CacheStats TensorCache::stats() const { std::lock_guard lk(mu_); return stats_; }
std::uint64_t TensorCache::used_bytes() const { std::lock_guard lk(mu_); return used_; }
}
