#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace memvanta {

inline std::size_t checked_add_size(std::size_t a,std::size_t b,const char* what){
    if(b>std::numeric_limits<std::size_t>::max()-a) throw std::runtime_error(what);
    return a+b;
}

inline std::size_t checked_mul_size(std::size_t a,std::size_t b,const char* what){
    if(a && b>std::numeric_limits<std::size_t>::max()/a) throw std::runtime_error(what);
    return a*b;
}

inline std::size_t checked_bytes_for_floats(std::size_t count,const char* what){
    return checked_mul_size(count,sizeof(float),what);
}

inline std::size_t checked_bytes_for_u16(std::size_t count,const char* what){
    return checked_mul_size(count,sizeof(std::uint16_t),what);
}

} // namespace memvanta
