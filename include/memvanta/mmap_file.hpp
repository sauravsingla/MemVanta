#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
namespace memvanta {
struct MMapAdviceStats {
  std::uint64_t attempts{};
  std::uint64_t failures{};
};
class MMapFile {
public:
  MMapFile() = default;
  explicit MMapFile(const std::string& path);
  ~MMapFile();
  MMapFile(const MMapFile&) = delete;
  MMapFile& operator=(const MMapFile&) = delete;
  MMapFile(MMapFile&&) noexcept;
  MMapFile& operator=(MMapFile&&) noexcept;
  void open(const std::string& path);
  void close();
  const std::byte* data() const { return data_; }
  std::uint64_t size() const { return size_; }
  bool valid() const { return data_ != nullptr; }
  void advise_sequential(std::uint64_t offset, std::uint64_t length) const;
  void advise_willneed(std::uint64_t offset, std::uint64_t length) const;
  void advise_dontneed(std::uint64_t offset, std::uint64_t length) const;
  MMapAdviceStats advice_stats() const { return {advice_attempts_.load(),advice_failures_.load()}; }
private:
  void advise_range(std::uint64_t offset,std::uint64_t length,int advice) const;
  int fd_ = -1;
  std::byte* data_ = nullptr;
  std::uint64_t size_ = 0;
  mutable std::atomic<std::uint64_t> advice_attempts_{0};
  mutable std::atomic<std::uint64_t> advice_failures_{0};
};
}
