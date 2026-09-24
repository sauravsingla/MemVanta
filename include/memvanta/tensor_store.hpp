#pragma once
#include "memvanta/mmap_file.hpp"

#include <cstdint>
#include <string>
#include <vector>
namespace memvanta {
struct TensorSlice {
    std::uint32_t id;
    std::uint64_t offset;
    std::uint64_t bytes;
};
class TensorStore {
  public:
    TensorStore(const std::string& path, std::uint64_t chunk_bytes);
    const TensorSlice& slice(std::uint32_t id) const { return slices_.at(id); }
    std::uint32_t count() const { return static_cast<std::uint32_t>(slices_.size()); }
    const std::byte* ptr(std::uint32_t id) const;
    void prefetch(std::uint32_t id) const;
    void release(std::uint32_t id) const;
    std::uint64_t file_size() const { return file_.size(); }
    std::uint64_t chunk_bytes() const { return chunk_bytes_; }

  private:
    MMapFile file_;
    std::uint64_t chunk_bytes_;
    std::vector<TensorSlice> slices_;
};
} // namespace memvanta
