#include "memvanta/gguf.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace memvanta {
namespace {
class Reader {
public:
    Reader(const std::byte* p, std::uint64_t n): p_(p), n_(n) {}
    std::uint64_t pos() const { return off_; }
    std::uint64_t remaining() const { return n_-off_; }
    template<class T> T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require(sizeof(T)); T v{}; std::memcpy(&v,p_+off_,sizeof(T)); off_ += sizeof(T); return v;
    }
    std::string str() {
        auto len=read<std::uint64_t>(); require(len);
        if(len>std::numeric_limits<std::size_t>::max()) throw std::runtime_error("GGUF string too large");
        std::string s(reinterpret_cast<const char*>(p_+off_), static_cast<std::size_t>(len)); off_ += len; return s;
    }
    void require(std::uint64_t k) const { if (k > n_ || off_ > n_-k) throw std::runtime_error("truncated GGUF"); }
private:
    const std::byte* p_{}; std::uint64_t n_{}; std::uint64_t off_{};
};

std::uint64_t checked_mul(std::uint64_t a,std::uint64_t b,const char* what){if(a&&b>std::numeric_limits<std::uint64_t>::max()/a)throw std::runtime_error(what);return a*b;}
std::uint64_t quant_row_block(GgmlType type){
    switch(type){
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q8_0: return 32;
        case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K: return 256;
        default: return 1;
    }
}

std::uint64_t as_unsigned(const GgufValue& v) {
    if (auto p=std::get_if<std::uint64_t>(&v.data)) return *p;
    if (auto p=std::get_if<std::int64_t>(&v.data)) { if (*p<0) throw std::runtime_error("negative metadata value"); return static_cast<std::uint64_t>(*p); }
    throw std::runtime_error("metadata value is not integer");
}
GgufValue read_scalar(Reader& r, GgufValueType t) {
    GgufValue v; v.type=t;
    switch(t) {
        case GgufValueType::UInt8: v.data=static_cast<std::uint64_t>(r.read<std::uint8_t>()); break;
        case GgufValueType::Int8: v.data=static_cast<std::int64_t>(r.read<std::int8_t>()); break;
        case GgufValueType::UInt16: v.data=static_cast<std::uint64_t>(r.read<std::uint16_t>()); break;
        case GgufValueType::Int16: v.data=static_cast<std::int64_t>(r.read<std::int16_t>()); break;
        case GgufValueType::UInt32: v.data=static_cast<std::uint64_t>(r.read<std::uint32_t>()); break;
        case GgufValueType::Int32: v.data=static_cast<std::int64_t>(r.read<std::int32_t>()); break;
        case GgufValueType::Float32: v.data=static_cast<double>(r.read<float>()); break;
        case GgufValueType::Bool: v.data=static_cast<bool>(r.read<std::uint8_t>() != 0); break;
        case GgufValueType::String: v.data=r.str(); break;
        case GgufValueType::UInt64: v.data=r.read<std::uint64_t>(); break;
        case GgufValueType::Int64: v.data=r.read<std::int64_t>(); break;
        case GgufValueType::Float64: v.data=r.read<double>(); break;
        case GgufValueType::Array: throw std::runtime_error("nested GGUF arrays are unsupported");
        default: throw std::runtime_error("unknown GGUF metadata type");
    }
    return v;
}
GgufValue read_value(Reader& r, GgufValueType t) {
    if (t != GgufValueType::Array) return read_scalar(r,t);
    const auto elem=static_cast<GgufValueType>(r.read<std::uint32_t>());
    if(elem==GgufValueType::Array) throw std::runtime_error("nested GGUF arrays are unsupported");
    const auto n=r.read<std::uint64_t>();
    if(n>std::numeric_limits<std::size_t>::max()) throw std::runtime_error("GGUF array too large");
    GgufValue v; v.type=GgufValueType::Array;
    if (elem==GgufValueType::String) {
        if(n>r.remaining()/sizeof(std::uint64_t)) throw std::runtime_error("truncated GGUF string array");
        GgufValue::StringArray a; a.reserve(static_cast<std::size_t>(n));
        for(std::uint64_t i=0;i<n;++i) a.push_back(r.str()); v.data=std::move(a); return v;
    }
    if (elem==GgufValueType::Float32 || elem==GgufValueType::Float64) {
        const std::uint64_t width=elem==GgufValueType::Float32?sizeof(float):sizeof(double);
        if(n>r.remaining()/width) throw std::runtime_error("truncated GGUF float array");
        GgufValue::FloatArray a; a.reserve(static_cast<std::size_t>(n));
        for(std::uint64_t i=0;i<n;++i) a.push_back(elem==GgufValueType::Float32 ? static_cast<double>(r.read<float>()) : r.read<double>());
        v.data=std::move(a); return v;
    }
    std::uint64_t width=0;
    switch(elem){
        case GgufValueType::UInt8: case GgufValueType::Int8: case GgufValueType::Bool: width=1; break;
        case GgufValueType::UInt16: case GgufValueType::Int16: width=2; break;
        case GgufValueType::UInt32: case GgufValueType::Int32: width=4; break;
        case GgufValueType::UInt64: case GgufValueType::Int64: width=8; break;
        default: throw std::runtime_error("unsupported GGUF array element type");
    }
    if(n>r.remaining()/width) throw std::runtime_error("truncated GGUF integer array");
    GgufValue::IntArray a; a.reserve(static_cast<std::size_t>(n));
    for(std::uint64_t i=0;i<n;++i) {
        auto sv=read_scalar(r,elem);
        if(auto p=std::get_if<std::uint64_t>(&sv.data)){if(*p>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))throw std::runtime_error("GGUF integer array value exceeds int64");a.push_back(static_cast<std::int64_t>(*p));}
        else if(auto p=std::get_if<std::int64_t>(&sv.data)) a.push_back(*p);
        else if(auto p=std::get_if<bool>(&sv.data)) a.push_back(*p?1:0);
        else throw std::runtime_error("unsupported GGUF array element type");
    }
    v.data=std::move(a); return v;
}
}

std::uint64_t GgufTensor::elements() const {
    std::uint64_t n=1; for(auto d:dims){ if(!d) throw std::runtime_error("zero tensor dimension"); if(n>std::numeric_limits<std::uint64_t>::max()/d) throw std::runtime_error("tensor element overflow"); n*=d; } return n;
}

std::uint64_t ggml_tensor_nbytes(GgmlType type, std::uint64_t n) {
    auto blocks=[&](std::uint64_t qk,std::uint64_t bs){ if(n%qk) throw std::runtime_error("quantized tensor element count is not block aligned"); return checked_mul(n/qk,bs,"tensor byte-size overflow"); };
    switch(type) {
        case GgmlType::F32: return checked_mul(n,4,"tensor byte-size overflow");
        case GgmlType::F16: return checked_mul(n,2,"tensor byte-size overflow");
        case GgmlType::Q4_0: return blocks(32,18);
        case GgmlType::Q4_1: return blocks(32,20);
        case GgmlType::Q5_0: return blocks(32,22);
        case GgmlType::Q5_1: return blocks(32,24);
        case GgmlType::Q8_0: return blocks(32,34);
        case GgmlType::I8: return n;
        case GgmlType::I16: return checked_mul(n,2,"tensor byte-size overflow");
        case GgmlType::I32: return checked_mul(n,4,"tensor byte-size overflow");
        case GgmlType::I64: return checked_mul(n,8,"tensor byte-size overflow");
        case GgmlType::F64: return checked_mul(n,8,"tensor byte-size overflow");
        case GgmlType::BF16: return checked_mul(n,2,"tensor byte-size overflow");
        case GgmlType::Q2_K: return blocks(256,84);
        case GgmlType::Q3_K: return blocks(256,110);
        case GgmlType::Q4_K: return blocks(256,144);
        case GgmlType::Q5_K: return blocks(256,176);
        case GgmlType::Q6_K: return blocks(256,210);
        default: throw std::runtime_error("unsupported GGML type for byte-size calculation: "+std::to_string(static_cast<std::uint32_t>(type)));
    }
}
std::string ggml_type_name(GgmlType t) {
    switch(t){case GgmlType::F32:return"F32";case GgmlType::F16:return"F16";case GgmlType::Q4_0:return"Q4_0";case GgmlType::Q4_1:return"Q4_1";case GgmlType::Q5_0:return"Q5_0";case GgmlType::Q5_1:return"Q5_1";case GgmlType::Q8_0:return"Q8_0";case GgmlType::Q2_K:return"Q2_K";case GgmlType::Q3_K:return"Q3_K";case GgmlType::Q4_K:return"Q4_K";case GgmlType::Q5_K:return"Q5_K";case GgmlType::Q6_K:return"Q6_K";default:return"type-"+std::to_string(static_cast<std::uint32_t>(t));}
}

float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t s=(h>>15)&1u, e=(h>>10)&0x1fu, f=h&0x3ffu;
    std::uint32_t out;
    if(e==0){
        if(f==0) out=s<<31;
        else { std::uint32_t ff=f, ee=127-15+1; while((ff&0x400u)==0){ff<<=1;--ee;} ff&=0x3ffu; out=(s<<31)|(ee<<23)|(ff<<13); }
    } else if(e==31) out=(s<<31)|0x7f800000u|(f<<13);
    else out=(s<<31)|((e+(127-15))<<23)|(f<<13);
    return std::bit_cast<float>(out);
}

GgufFile::GgufFile(const std::string& path): file_(path) {
    Reader r(file_.data(),file_.size());
    char magic[5]{}; for(int i=0;i<4;++i) magic[i]=static_cast<char>(r.read<std::uint8_t>());
    if(std::string(magic,4)!="GGUF") throw std::runtime_error("not a GGUF file");
    version_=r.read<std::uint32_t>(); if(version_<2 || version_>3) throw std::runtime_error("unsupported GGUF version: "+std::to_string(version_));
    const auto nt=r.read<std::uint64_t>(), nkv=r.read<std::uint64_t>();
    if(nt>1000000 || nkv>1000000) throw std::runtime_error("implausible GGUF counts");
    for(std::uint64_t i=0;i<nkv;++i){ auto key=r.str(); auto type=static_cast<GgufValueType>(r.read<std::uint32_t>()); auto value=read_value(r,type); if(!metadata_.emplace(std::move(key),std::move(value)).second) throw std::runtime_error("duplicate GGUF metadata key"); }
    struct Tmp{std::string name;std::vector<std::uint64_t>dims;GgmlType type;std::uint64_t rel;};
    std::vector<Tmp> tmp; tmp.reserve(static_cast<std::size_t>(nt));
    for(std::uint64_t i=0;i<nt;++i){ Tmp t; t.name=r.str(); auto nd=r.read<std::uint32_t>(); if(nd==0||nd>4) throw std::runtime_error("invalid tensor dimensions"); t.dims.resize(nd); for(auto&d:t.dims){d=r.read<std::uint64_t>();if(!d)throw std::runtime_error("zero tensor dimension: "+t.name);} t.type=static_cast<GgmlType>(r.read<std::uint32_t>()); t.rel=r.read<std::uint64_t>(); tmp.push_back(std::move(t)); }
    std::uint64_t align=32; if(auto it=metadata_.find("general.alignment");it!=metadata_.end()) align=as_unsigned(it->second); if(!align || (align&(align-1))) throw std::runtime_error("invalid GGUF alignment");
    if(r.pos()>std::numeric_limits<std::uint64_t>::max()-(align-1)) throw std::runtime_error("GGUF data offset overflow");
    data_offset_=(r.pos()+align-1)&~(align-1);
    if(nt && data_offset_>file_.size()) throw std::runtime_error("GGUF data section starts beyond file bounds");
    tensors_.reserve(tmp.size());
    for(auto&t:tmp){ GgufTensor x; x.name=std::move(t.name); x.dims=std::move(t.dims); x.type=t.type; const auto qk=quant_row_block(x.type); if(qk>1&&x.ne(0)%qk)throw std::runtime_error("quantized tensor row is not block aligned: "+x.name); if(t.rel>std::numeric_limits<std::uint64_t>::max()-data_offset_) throw std::runtime_error("tensor offset overflow: "+x.name); x.offset=data_offset_+t.rel; x.nbytes=ggml_tensor_nbytes(x.type,x.elements()); if(x.offset>file_.size()||x.nbytes>file_.size()-x.offset) throw std::runtime_error("tensor out of file bounds: "+x.name); if(tensor_index_.contains(x.name)) throw std::runtime_error("duplicate GGUF tensor: "+x.name); tensor_index_[x.name]=tensors_.size(); tensors_.push_back(std::move(x)); }
}
const GgufTensor& GgufFile::tensor(const std::string& name) const { auto it=tensor_index_.find(name); if(it==tensor_index_.end()) throw std::runtime_error("missing tensor: "+name); return tensors_[it->second]; }
bool GgufFile::has_tensor(const std::string& name) const { return tensor_index_.contains(name); }
const std::byte* GgufFile::tensor_data(const GgufTensor& t) const { return file_.data()+t.offset; }
std::optional<std::uint64_t> GgufFile::get_u64(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return{};try{return as_unsigned(it->second);}catch(...){return{};}}
std::optional<std::int64_t> GgufFile::get_i64(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return{};if(auto p=std::get_if<std::int64_t>(&it->second.data))return*p;if(auto p=std::get_if<std::uint64_t>(&it->second.data)){if(*p>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))return{};return static_cast<std::int64_t>(*p);}return{};}
std::optional<double> GgufFile::get_f64(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return{};if(auto p=std::get_if<double>(&it->second.data))return*p;if(auto p=std::get_if<std::uint64_t>(&it->second.data))return static_cast<double>(*p);if(auto p=std::get_if<std::int64_t>(&it->second.data))return static_cast<double>(*p);return{};}
std::optional<bool> GgufFile::get_bool(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return{};if(auto p=std::get_if<bool>(&it->second.data))return*p;return{};}
std::optional<std::string> GgufFile::get_string(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return{};if(auto p=std::get_if<std::string>(&it->second.data))return*p;return{};}
const std::vector<std::string>* GgufFile::get_string_array(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return nullptr;return std::get_if<GgufValue::StringArray>(&it->second.data);}
const std::vector<double>* GgufFile::get_float_array(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return nullptr;return std::get_if<GgufValue::FloatArray>(&it->second.data);}
const std::vector<std::int64_t>* GgufFile::get_int_array(const std::string& k) const {auto it=metadata_.find(k);if(it==metadata_.end())return nullptr;return std::get_if<GgufValue::IntArray>(&it->second.data);}

} // namespace memvanta
