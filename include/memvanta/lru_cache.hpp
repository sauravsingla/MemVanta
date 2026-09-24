#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace memvanta {

struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bytes_copied = 0;
};

class TensorCache {
  public:
    explicit TensorCache(std::uint64_t capacity_bytes);

    std::shared_ptr<const std::vector<std::byte>>
    get_or_load(std::uint32_t id, const std::byte* src, std::uint64_t bytes);
    bool contains(std::uint32_t id) const;
    void insert_prefetched(std::uint32_t id, const std::byte* src, std::uint64_t bytes);
    CacheStats stats() const;
    std::uint64_t used_bytes() const;
    std::uint64_t prefetched_bytes() const;
    std::uint64_t peak_prefetched_bytes() const;

  private:
    struct Entry {
        std::shared_ptr<std::vector<std::byte>> data;
        std::list<std::uint32_t>::iterator lru_it;
        bool prefetched = false;
    };

    void evict_for(std::uint64_t incoming);

    std::uint64_t capacity_;
    std::uint64_t used_ = 0;
    std::uint64_t prefetched_used_ = 0;
    std::uint64_t peak_prefetched_used_ = 0;
    mutable std::mutex mu_;
    std::list<std::uint32_t> lru_;
    std::unordered_map<std::uint32_t, Entry> map_;
    CacheStats stats_;
};

} // namespace memvanta
