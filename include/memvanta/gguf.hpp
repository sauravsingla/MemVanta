#pragma once
#include "memvanta/mmap_file.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace memvanta {

enum class GgufValueType : std::uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12
};

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30
};

struct GgufValue {
    using IntArray = std::vector<std::int64_t>;
    using FloatArray = std::vector<double>;
    using StringArray = std::vector<std::string>;
    using Data = std::variant<std::uint64_t,
                              std::int64_t,
                              double,
                              bool,
                              std::string,
                              IntArray,
                              FloatArray,
                              StringArray>;
    GgufValueType type{};
    Data data{};
};

struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> dims; // GGML order: ne0, ne1, ...
    GgmlType type{};
    std::uint64_t offset{}; // absolute file offset
    std::uint64_t nbytes{};
    std::uint64_t elements() const;
    std::uint64_t ne(std::size_t i) const { return i < dims.size() ? dims[i] : 1; }
};

struct GgufParseLimits {
    std::uint64_t max_tensors{1'000'000};
    std::uint64_t max_metadata_items{1'000'000};
    std::uint64_t max_array_elements{1'000'000};
    std::uint64_t max_string_bytes{64ull << 20};
    // Approximate heap budget for decoded metadata, tensor descriptors and parser
    // containers. Tensor payloads remain mmap-backed and are not charged here.
    std::uint64_t max_parse_heap_bytes{256ull << 20};
};

class GgufFile {
  public:
    explicit GgufFile(const std::string& path, GgufParseLimits limits = {});

    std::uint32_t version() const { return version_; }
    std::uint64_t data_offset() const { return data_offset_; }
    const MMapFile& mapped_file() const { return file_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    const GgufTensor& tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    const std::byte* tensor_data(const GgufTensor& t) const;

    std::optional<std::uint64_t> get_u64(const std::string& key) const;
    std::optional<std::int64_t> get_i64(const std::string& key) const;
    std::optional<double> get_f64(const std::string& key) const;
    std::optional<bool> get_bool(const std::string& key) const;
    std::optional<std::string> get_string(const std::string& key) const;
    const std::vector<std::string>* get_string_array(const std::string& key) const;
    const std::vector<double>* get_float_array(const std::string& key) const;
    const std::vector<std::int64_t>* get_int_array(const std::string& key) const;
    const std::unordered_map<std::string, GgufValue>& metadata() const { return metadata_; }

  private:
    MMapFile file_;
    std::uint32_t version_{};
    std::uint64_t data_offset_{};
    std::unordered_map<std::string, GgufValue> metadata_;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
};

std::uint64_t ggml_tensor_nbytes(GgmlType type, std::uint64_t elements);
std::string ggml_type_name(GgmlType type);
float fp16_to_fp32(std::uint16_t h);

} // namespace memvanta
