#include "memvanta/tensor_store.hpp"

#include <algorithm>
#include <stdexcept>
namespace memvanta {
TensorStore::TensorStore(const std::string& path, std::uint64_t chunk_bytes)
    : file_(path), chunk_bytes_(chunk_bytes) {
    if (!chunk_bytes_)
        throw std::runtime_error("chunk size must be > 0");
    std::uint64_t off = 0;
    std::uint32_t id = 0;
    while (off < file_.size()) {
        auto n = std::min(chunk_bytes_, file_.size() - off);
        slices_.push_back({id++, off, n});
        off += n;
    }
}
const std::byte* TensorStore::ptr(std::uint32_t id) const {
    auto& s = slice(id);
    return file_.data() + s.offset;
}
void TensorStore::prefetch(std::uint32_t id) const {
    auto& s = slice(id);
    file_.advise_willneed(s.offset, s.bytes);
}
void TensorStore::release(std::uint32_t id) const {
    auto& s = slice(id);
    file_.advise_dontneed(s.offset, s.bytes);
}
} // namespace memvanta
